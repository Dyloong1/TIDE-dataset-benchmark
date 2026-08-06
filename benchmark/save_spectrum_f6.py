"""save_spectrum_f6.py — F6 figure data: shell-averaged E(k) of AR-rollout prediction vs truth.

F6 (spectral-fidelity comparison / over-smoothing evidence) needs the shell-averaged energy spectrum E(k) of the
model's AR-rollout prediction at the final step (s19) compared against the ground-truth field's
E(k). NN rollout tends to over-smooth (lose high-k energy) as error accumulates — this figure
shows it directly.

DESIGN: this is a STANDALONE script that REUSES eval_benchmark's production AR-rollout logic
(_ar_rollout_streaming) and BM.energy_spectrum by import — it does NOT modify any shared eval
script, so the rollout scoring pipeline and its polarity/metric contracts are byte-identical to
the leaderboard eval. It only adds: after the rollout produces (last_pred, last_true), compute
each one's E(k) and save both curves to npz. GPU: one AR-20 rollout per sample (same cost as the
leaderboard eval's rollout). Output: mat_F6_spectrum_<case>.npz with k, E_pred, E_true per sample
+ their means. Averaged over N test samples for a clean figure.

USAGE (mirrors eval_benchmark's rollout invocation):
  python save_spectrum_f6.py --model transolver --task rollout --case ou_relam70_256_fp64 \
    --slice docs/slice/benchmark_slice.json --data-root C:/Yilong/turbgen_data \
    --ckpt checkpoints/bench_3/transolver__rollout__ou_relam70_256_fp64/best.pt --residual \
    --out paper_kdd2027/fig_materials_20260716/incoming_CYB-3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import solver._env  # noqa: F401,E402  (Windows OpenMP guard before torch)
import torch  # noqa: E402
import numpy as np  # noqa: E402

from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset  # noqa: E402
from Model.baselines import get_model  # noqa: E402
import benchmark_metrics as BM  # noqa: E402
from eval_benchmark import _ar_rollout_streaming  # noqa: E402  (reuse production AR logic)
from train_benchmark import build_config  # noqa: E402  (shared config wiring, model matches ckpt)


def _spectrum(field):
    """field (C,Z,H,W) velocity-only -> (k_int, E(k)) via production BM.energy_spectrum.
    energy_spectrum expects (3,Z,H,W) velocity; we pass channels 0..2 (u,v,w)."""
    u = field[:3] if field.shape[0] >= 3 else field
    k, e = BM.energy_spectrum(u)
    return k.cpu().numpy(), e.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", default="rollout", choices=("rollout",))  # F6 is a rollout figure
    ap.add_argument("--case", required=True)
    ap.add_argument("--slice", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--t-in", type=int, default=1)
    ap.add_argument("--t-out", type=int, default=20)     # AR-20, matches eval protocol
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--device", default="cuda")
    # HP fallbacks read by build_config (same defaults as eval_benchmark); production_configs_256.json
    # overrides these per-model so the built model matches the ckpt.
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--tfno-rank", type=int, default=8)
    ap.add_argument("--deeponet-p", type=int, default=128)
    args = ap.parse_args()

    dev = torch.device(args.device)
    ds = TurbgenZarrDataset(case=args.case, split="test", slice_manifest=args.slice,
                            task="rollout", t_in=args.t_in, t_out=args.t_out,
                            data_root=args.data_root, patch_size=None, normalize=True)
    # build model via the SAME production config wiring eval_benchmark uses (so the model
    # structure matches the ckpt: native 256 grid, INDICATORS channels, t_in/t_out).
    # build_config pre-seeds the FNO-family keys from args.* and only setdefault()s the JSON on top,
    # so the CLI value WINS. To match the ckpt we seed args.* from the model's production JSON first
    # (FNO_MODES_Z -> --modes etc.), exactly the per-model capacity the suite trained with.
    from train_benchmark import _arch_keys_from_json  # noqa: E402
    jk = _arch_keys_from_json(args.model)
    if jk.get("FNO_MODES_Z") is not None:
        args.modes = int(jk["FNO_MODES_Z"])
    if jk.get("FNO_WIDTH") is not None:
        args.width = int(jk["FNO_WIDTH"])
    if jk.get("FNO_LAYERS") is not None:
        args.layers = int(jk["FNO_LAYERS"])
    if jk.get("TFNO_RANK") is not None:
        args.tfno_rank = int(jk["TFNO_RANK"])
    in_ch = out_ch = len(getattr(ds, "indicators", ["u", "v", "w", "p"]))
    cfg = build_config(args, in_ch, out_ch, args.t_in, 1, model=args.model, grid_n=256)
    model = get_model(args.model, cfg)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck.get("model", ck))
        model.load_state_dict(sd)
    model = model.to(dev).eval()

    n_total = len(ds)
    idxs = list(range(0, n_total, max(1, n_total // args.samples)))[: args.samples]
    print(f"[F6] {args.case} {args.model}: {len(idxs)}/{n_total} samples, AR-{args.t_out}")

    E_preds, E_trues, kref = [], [], None
    for i, idx in enumerate(idxs):
        s = ds[idx]
        x = s["input_dense"].unsqueeze(0).to(dev)
        y = s["output_dense"].unsqueeze(0)  # keep on CPU; rollout moves per-frame
        with torch.no_grad():
            _per, last_pred, last_true = _ar_rollout_streaming(
                model, x, y, k0=None, decay=False, residual=args.residual, dev=dev)
        kp, ep = _spectrum(last_pred)
        kt, et = _spectrum(last_true)
        kref = kp
        E_preds.append(ep)
        E_trues.append(et)
        print(f"  sample {i+1}/{len(idxs)} (idx {idx}): E_pred[1]={ep[1]:.3e} E_true[1]={et[1]:.3e}")

    E_pred_mean = np.mean(E_preds, axis=0)
    E_true_mean = np.mean(E_trues, axis=0)
    os.makedirs(args.out, exist_ok=True)
    outp = os.path.join(args.out, f"mat_F6_spectrum_{args.case}_{args.model}.npz")
    import subprocess
    try:
        gh = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        gh = "unknown"
    np.savez(outp, k=kref, E_pred_mean=E_pred_mean, E_true_mean=E_true_mean,
             E_pred_all=np.array(E_preds), E_true_all=np.array(E_trues),
             provenance=(f"F6 spectrum-fidelity {args.case} {args.model} | AR-{args.t_out} last-frame "
                         f"E(k) pred vs truth | {len(idxs)} test samples mean | BM.energy_spectrum "
                         f"(production) | reuses eval_benchmark._ar_rollout_streaming (byte-identical "
                         f"rollout) | git={gh}"))
    print(f"[F6] wrote {outp}")
    # F6 signal summary. Two failure modes a NN rollout shows in the spectrum, read off the
    # resolved band (not the far tail): the far tail k>=~86 sits at DNS truth ~1e-9..1e-12
    # (machine-zero near k_max*eta), so a ratio there just divides by ~0 and is meaningless.
    # Report on the mid-band (k in [8, 32], energy-containing to inertial) instead:
    #   ratio < 1  -> over-smoothing (NN loses energy, the classic diffusion blur)
    #   ratio > 1  -> spurious energy injection (AR high-freq garbage climbing into resolved k)
    lo, hi = 8, min(32, len(kref) - 1)
    mid = slice(lo, hi)
    ratio = float(E_pred_mean[mid].sum() / (E_true_mean[mid].sum() + 1e-30))
    tag = "over-smoothed" if ratio < 0.8 else ("spurious-hi-energy" if ratio > 1.25 else "ok")
    print(f"[F6] mid-band k[{lo}:{hi}] energy ratio pred/truth = {ratio:.3f} ({tag})")
    # far-tail note (informational only): NN injects small-absolute but relatively huge hi-k noise
    ft = slice(len(kref) * 2 // 3, None)
    print(f"[F6] (far-tail k>={kref[len(kref)*2//3]}: truth~{E_true_mean[ft].sum():.1e} is DNS "
          f"machine-zero; NN adds {E_pred_mean[ft].sum():.1e} spurious hi-k energy — see curve, not ratio)")


if __name__ == "__main__":
    main()
