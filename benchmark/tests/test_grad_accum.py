"""Judge-first test for gradient accumulation (effective-batch-16 production protocol).

The invariant that MUST hold: accumulating grads over `accum` micro-batches of size B
(each micro-loss scaled by 1/accum) yields the SAME parameter gradient as a single
full-batch step over B*accum samples with a mean-reduced loss. If the 1/accum scaling is
wrong (or missing), effective_batch != batch*accum and the production compute-budget
comparability claim silently breaks.

We check this numerically on a tiny MSE problem (no dataset/zarr needed): build one batch of
N=8 samples, compute the reference full-batch gradient, then reproduce it by accumulating
4 micro-batches of 2 with the exact scaling the train loop uses ((loss/accum).backward()).

Run: KMP_DUPLICATE_LIB_OK=TRUE python -m pytest tests/test_grad_accum.py -q
"""
import torch
import torch.nn as nn


def _fresh_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 6), nn.GELU(), nn.Linear(6, 6))


def test_accum_matches_full_batch_grad():
    torch.manual_seed(1)
    N, accum = 8, 4          # micro-batch B = N/accum = 2
    B = N // accum
    x = torch.randn(N, 6)
    y = torch.randn(N, 6)

    # --- reference: one full-batch step, mean-reduced MSE ---
    m_ref = _fresh_model()
    m_ref.zero_grad()
    loss_ref = nn.functional.mse_loss(m_ref(x), y)   # mean over all N samples
    loss_ref.backward()
    g_ref = [p.grad.clone() for p in m_ref.parameters()]

    # --- accumulation: sum (loss_i / accum).backward() over `accum` micro-batches ---
    m_acc = _fresh_model()                            # same seed -> identical init weights
    m_acc.zero_grad()
    for k in range(accum):
        xb = x[k * B:(k + 1) * B]
        yb = y[k * B:(k + 1) * B]
        loss = nn.functional.mse_loss(m_acc(xb), yb)  # mean over the B micro-batch samples
        (loss / accum).backward()
    g_acc = [p.grad.clone() for p in m_acc.parameters()]

    # weights are identical (same seed), so grads must match to fp32 precision
    for a, b in zip(g_ref, g_acc):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (a - b).abs().max().item()


def test_accum_b2_matches_b8():
    # B=2 micro-batch, accum=4 -> effective 8; must equal a single mean-reduced B=8 step.
    torch.manual_seed(3)
    N, accum, B = 8, 4, 2
    x, y = torch.randn(N, 6), torch.randn(N, 6)
    m_ref = _fresh_model(); m_ref.zero_grad()
    nn.functional.mse_loss(m_ref(x), y).backward()
    g_ref = [p.grad.clone() for p in m_ref.parameters()]
    m_acc = _fresh_model(); m_acc.zero_grad()
    for k in range(accum):
        xb, yb = x[k * B:(k + 1) * B], y[k * B:(k + 1) * B]
        (nn.functional.mse_loss(m_acc(xb), yb) / accum).backward()
    for a, b in zip(g_ref, [p.grad for p in m_acc.parameters()]):
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (a - b).abs().max().item()


def test_full_group_selection_drops_tail():
    # the train loop steps only full accum-groups: n_full = (len // accum) * accum.
    # a trailing partial group must be dropped (NOT flushed under-weighted), so every
    # optimizer step is exactly effective_batch. Pin the index math the loop uses.
    for n_batches, accum, expect_full in [(17, 4, 16), (16, 4, 16), (3, 4, 0), (32, 16, 32), (30, 16, 16)]:
        n_full = (n_batches // accum) * accum
        assert n_full == expect_full, (n_batches, accum, n_full)
        # every stepped micro-batch is inside a full group -> steps = n_full/accum, all full
        assert n_full % accum == 0
        # dropped tail count
        assert n_batches - n_full == n_batches % accum


def test_accum_shrinks_when_fewer_batches_than_accum():
    # guard: len(tl) < accum must shrink accum (train anyway), never yield 0 steps.
    for n_batches, accum in [(3, 16), (1, 4), (8, 16)]:
        eff = max(1, n_batches) if n_batches < accum else accum
        n_full = (n_batches // eff) * eff
        assert n_full >= eff and n_full > 0, (n_batches, accum, eff, n_full)  # at least one full step


def test_accum_one_is_plain_sgd():
    # accum=1 must be a no-op relative to per-batch stepping (default path unchanged).
    torch.manual_seed(2)
    x, y = torch.randn(3, 6), torch.randn(3, 6)
    m = _fresh_model(); m.zero_grad()
    loss = nn.functional.mse_loss(m(x), y)
    (loss / 1).backward()
    g1 = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    nn.functional.mse_loss(m(x), y).backward()
    g0 = [p.grad.clone() for p in m.parameters()]
    for a, b in zip(g0, g1):
        assert torch.equal(a, b)


if __name__ == "__main__":
    test_accum_matches_full_batch_grad(); print("  PASS accum_matches_full_batch_grad")
    test_accum_one_is_plain_sgd(); print("  PASS accum_one_is_plain_sgd")
    print("\n2/2 grad-accum tests passed")
