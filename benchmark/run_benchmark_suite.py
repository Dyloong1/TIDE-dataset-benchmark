"""run_benchmark_suite.py — resumable orchestrator for the q-side benchmark full run.

Drives train_benchmark.py + eval_benchmark.py over the (model x task x case) matrix at
run-level GRANULARITY, so a crash/OOM/reboot loses at most the one run in flight:

  * per-run dir  <root>/<model>__<task>__<case>/ holds status.json + train.log + eval.log
    + the model's best.pt/latest.pt (train_benchmark saves latest.pt every epoch).
  * status.json is written IMMEDIATELY at each transition (queued->train_running->trained->
    eval_running->done, or *_failed) — real-time, not batched at suite end.
  * RESUME: on restart the orchestrator SKIPS any run whose status.json says done (idempotent);
    failed/partial runs re-run. Pass --redo to force-rerun everything.

Matrix = design doc section 4 minimal required sets (P1-P5 main table). rollout uses the AR
protocol: ONE frame in (NS is Markovian -- u(t) is a complete state, so an input history is pure
redundancy), TRAIN single-step (--t-out 1), EVAL autoregressive unroll 20 steps (1.0 T_L @ dt=0.05).
The design variable is the frame INTERVAL (0.05 T_L), not the input length. Per-model
hyperparams + physical batch come from production_configs_256.json; effective_batch is pinned
to 16 via gradient accumulation (accum = 16 / physical_batch).

  python run_benchmark_suite.py --root checkpoints/bench_q --data-root "$TURBGEN_DATA_DIR" \
      --slice "$TURBGEN_DATA_DIR/corpus/benchmark_slice.json" --epochs 25

Non-NN baselines are eval-only (no train): persistence/spectral_interp/identity/smagorinsky, plus
`poisson` (P4), which is not a floor but an EXACT spectral solution -- a ceiling on achievable
pressure accuracy.
solver/eval_dns/production untouched — this only calls the two benchmark CLIs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PROD_CFG = HERE / "Model" / "configs" / "production_configs_256.json"

# ---- training matrix: design doc section 4 minimal required sets (P1-P5) --------------------
# Each task -> the NN baselines that MUST run it (non-NN lower bounds are eval-only, added
# separately). re-curated 2026-07-13 (11->7 learned), then TRIMMED 2026-07-14 (schedule): rollout
# dropped dpot (it only appeared in rollout, which is ~93% of the wall-clock at 1.9h/run — cutting it
# saves ~13h and keeps all three families intact). Kept: FNO family fno3d+tfno, attention transolver,
# deeponet (resolution-bound); dpot stays importable in the registry as an appendix model (same as
# pde_refiner/acdm) but is not auto-run. pressure/sgs minimal set = channel-flexible fno3d/tfno/
# deeponet; the channel guard refuses transolver (out==in). EVERY model evals through the same
# 128^3-tiled forward stitched back to native 256^3 (_TiledModel) -- no per-model eval resolution.
REQUIRED = {
    # ufno (U-FNO, CNN/U-Net family) added 2026-07-14 per D&B-reviewer sim: it closes the "no
    # CNN/U-Net baseline" gap that PDEBench/The Well/PDEArena reviewers expect, AND breaks the
    # spectral-only monoculture on superres/sgs (which were fno3d-vs-tfno only), AND thickens the
    # thin 2-baseline tasks. Runs every task (separate in/out channels). dpot dropped same day
    # (family-redundant with transolver, schedule); net wall-clock ≈ flat (swap dpot->ufno).
    "rollout":  ["fno3d", "tfno", "transolver", "ufno"],  # FNO x2 + attention + CNN/U-Net
    "superres": ["fno3d", "tfno", "ufno"],                # +ufno breaks the FNO-vs-FNO leaderboard
    "recon":    ["tfno", "transolver", "ufno"],
    "pressure": ["fno3d", "tfno", "deeponet"],            # deeponet is the architecture diff point here
    "sgs":      ["tfno", "fno3d", "ufno"],                # +ufno breaks the FNO-vs-FNO leaderboard
}
# generative baselines (the "generative vs NO" comparison line) are NOT in the auto REQUIRED
# batch: they are appendix, run ONLY via explicit --models. Reasons the harness does not yet
# support them plug-and-play in rollout: (a) --residual is applied to all rollout models, but
# generative models train via get_training_loss (no delta) and eval does model(x)+last_frame ->
# systematic prediction drift (contract mismatch); (b) acdm is a 100-step DDPM sampler, so AR
# 30 x 100 x 32 samples x native-256^3 is astronomically slow / OOM; (c) build_config does not
# pass ACDM_*/REFINER_* hyperparams, so production_configs entries never take effect. Run these
# only after a dedicated generative harness (residual-off + sampler-step budget + hp wiring).
GENERATIVE_APPENDIX = ["pde_refiner", "acdm"]
# non-NN eval-only lower bounds per task (no training)
NONNN = {
    "rollout":  ["persistence"],
    "superres": ["spectral_interp"],
    "recon":    ["identity"],
    # P4 was the only task with NO non-NN reference, so a reader could not tell whether a low
    # pressure nRMSE meant the model learned the Poisson operator or merely fit the mean field.
    # `poisson` solves lap(p) = -d_i d_j(u_i u_j) spectrally via solver.operators.pressure_hat --
    # the same D4-certified operator that generated the corpus pressure channel. Unlike the other
    # entries this is an EXACT solution, so it is a ceiling on achievable accuracy rather than a
    # floor: it reproduces the corpus pressure to relative L2 1.2e-06 (corr 1.000000).
    "pressure": ["poisson"],
    "sgs":      ["smagorinsky"],
}
# q-disk in_dist configs (all trainable). decay/Re-axis/held-outs are t-disk / post-merge.
Q_CONFIGS = [
    "ou_robust_kf3_256_fp64", "ou_robust_kf4_256_fp64", "ou_robust_tau1_256_fp64",
    "rotating_ro0p2_256_fp64", "rotating_ro0p2_v2_256_fp64", "scalar_sc1_256_fp64",
]
EFFECTIVE_BATCH = 16


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load_models():
    return json.loads(PROD_CFG.read_text(encoding="utf-8"))["models"]


def _hp_args(model, models):
    """Model hyperparam CLI flags from production_configs (skip _meta fields)."""
    hp = models.get(model, {})
    flags = []
    seen = set()
    m = {"FNO_WIDTH": "--width", "FNO_LAYERS": "--layers",
         "FNO_MODES_Z": "--modes", "TFNO_RANK": "--tfno-rank", "DEEPONET_P": "--deeponet-p"}
    for k, flag in m.items():
        if k in hp and flag not in seen:  # --width/--modes only once
            flags += [flag, str(hp[k])]
            seen.add(flag)
    return flags


def _phys_batch(model, models):
    """Physical batch per model (VRAM-bound); effective_batch=16 via accum. Default 1."""
    return int(models.get(model, {}).get("_phys_batch", 1))


def _status_path(run_dir):
    return run_dir / "status.json"


def _read_status(run_dir):
    p = _status_path(run_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _write_status(run_dir, **kw):
    run_dir.mkdir(parents=True, exist_ok=True)
    st = _read_status(run_dir)
    st.update(kw); st["updated"] = _now()
    _status_path(run_dir).write_text(json.dumps(st, indent=2))


def _run(cmd, log_path):
    """Run a subprocess, tee stdout+stderr to log_path, return returncode."""
    env = dict(os.environ)
    # expandable_segments defrags the CUDA caching allocator's reserved pool. The AR-eval loop at
    # native 256^3 otherwise strands ~8GB as "reserved but unallocated" and OOMs heavier models
    # (tfno) even when the true forward fits. Required for the AR eval; harmless elsewhere.
    # ONLY add it if PYTORCH_CUDA_ALLOC_CONF is not set at all — respect a caller's explicit
    # config (Windows torch silently ignores expandable_segments, so t sets a Windows-compatible
    # allocator config that must not be clobbered by appending expandable_segments here).
    prev = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if not prev:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Unbuffered child stdout. Python block-buffers (4-8KB) when stdout is a FILE rather than a
    # tty, and we hand it `lf` below -- so a multi-hour training run wrote NOTHING to train.log
    # until it exited. Measured on q: 19 minutes into a live run (GPU allocated, 4.6 GB/12s being
    # read off the corpus, kernel parked in blk_mq_get_tag) the log still held only its 2-line
    # header. An operator watching that log sees a hung job; the only way to tell it apart from a
    # real hang was to poll /proc/<pid>/io by hand.
    # This is not cosmetic for a ~2.7-day, 126-run sweep: without it there is no way to see
    # progress, spot a stall, or know which epoch a killed run died on. CYB-3 hit the same thing
    # (train.log showed no progress because redirected Python stdout is buffered) and flagged it as an observability gap.
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w") as lf:
        lf.write(f"# {_now()}\n# {' '.join(cmd)}\n\n"); lf.flush()
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
    return p.returncode


def _train_frame_count(slice_path, case):
    """How many corpus frames the train seeds of `case` contribute — the exact GPU-cache size
    needed so every unique train frame is decompressed once into VRAM (~0.13GB/frame fp16 at 256^3).
    Reads the slice's per_config train seeds and their frames_in_corpus from the manifest side;
    falls back to a safe 100 if the manifest lacks per-seed counts."""
    try:
        sl = json.loads(Path(slice_path).read_text(encoding="utf-8"))
        pc = sl.get("per_config", sl)
        cc = pc.get(case, {})
        tr = (cc.get("train", {}) or {}).get("seeds", []) or []
        # sum the ACTUAL frames of the train seeds from the slice's `ranked` array (each entry
        # {seed, q, frames}); the old code read a `frames_per_seed` key that make_benchmark_slice
        # never writes, so it silently defaulted to 150 regardless of true trajectory length
        # (mis-sizing the GPU frame cache for short configs like scalar ~80 or relam90 ~250).
        ranked = {r["seed"]: r.get("frames") for r in (cc.get("ranked") or []) if "seed" in r}
        total = sum(int(ranked.get(s) or 150) for s in tr)   # fall back to 150 only if a seed's frames absent
        return max(1, total)
    except Exception:
        return 100


def _train_patch_for(model, args, models):
    """Per-model training-crop override (JSON `_train_patch`), else the suite-wide --patch.

    Exists for ufno: its multi-scale U-Net cannot fit a native-256^3 BACKWARD on 32GB even with
    per-block gradient checkpointing (Phase A 2026-07-16: OOM with and without GRAD_CKPT; the
    full-resolution skip tensors must persist for the decoder), so it keeps 128^3 crop training
    and relies on the FNO-family resolution transfer for its native-256^3 eval (fwd-only measured
    16.2GB, fits). That transfer claim is validated in Phase B, not assumed."""
    return int((models.get(model) or {}).get("_train_patch", args.patch))


def train_one(model, task, case, args, models):
    run_dir = Path(args.root) / f"{model}__{task}__{case}"
    st = _read_status(run_dir)
    if st.get("status") == "done" and not args.redo:
        print(f"  SKIP (done): {run_dir.name}")
        return "skipped"
    # skipped_oom: a permanent-skip marker written by the babysitter loop when a run kills the whole
    # scope on OOM more than once — the suite must NOT re-run it (else an infinite restart loop), and
    # it is NOT 'done' so it never enters the leaderboard. --redo clears it (retry deliberately).
    if st.get("status") == "skipped_oom" and not args.redo:
        print(f"  SKIP (oom, permanent): {run_dir.name}")
        return "skipped"
    _write_status(run_dir, model=model, task=task, case=case, status="train_running")

    # rollout: single-step training (t_out=1). superres/recon/pressure/sgs are frame maps -> also
    # t_out=1. eval expands rollout to t_out=20 via AR unroll (dt=0.05 T_L => 1.0 T_L horizon).
    #
    # t_in=1: incompressible NS is a first-order evolution equation, so the state u(t) is MARKOVIAN
    # -- u(t+dt) is fully determined by u(t) (pressure follows instantaneously from the Poisson
    # equation, it carries no time derivative). Feeding a history of 20 frames was 19 frames of pure
    # redundancy: it inflated every input tensor 20x (5.4GB per native-256 sample), made the loader
    # the throughput bottleneck, and forced a 50-frame window that left decay_hotstart with exactly
    # ONE usable rollout window. The only genuine design variable here is the time INTERVAL between
    # frames (0.05 T_L), not how many frames are stacked on the input.
    t_out_train = 1
    phys_b = _phys_batch(model, models)
    # Effective batch is defined in VOXELS, not samples, so the v1 crop protocol and the v2 native
    # protocol take identical optimizer steps: 16 x 128^3 crops == 2 x 256^3 fields == 33.6M voxels
    # per step. Keeping "16" at native would inflate the step to 8x the voxels AND leave only 27
    # steps/epoch — measured 2026-07-16: the epoch-based cosine then decays past the ~350-step
    # basin escape and starves the run (fno3d es=0/8, train still falling at lr=0).
    train_patch = _train_patch_for(model, args, models)
    if train_patch >= 256:
        # JSON _phys_batch was measured at 128^3 and is 8x too optimistic at native (fno3d
        # batch1 alone is 20.7GB with ckpt). Default to 1; a model may override with a
        # native-measured _phys_batch_native.
        phys_b = int((models.get(model) or {}).get("_phys_batch_native", 1))
        eff_batch = max(2, EFFECTIVE_BATCH // 8)
    else:
        eff_batch = EFFECTIVE_BATCH
    accum = max(1, eff_batch // phys_b)
    py = sys.executable
    # GPU-resident frame cache: decompress each unique train frame ONCE into VRAM (fp16, 0.13GB/
    # frame) so every epoch reads+crops on-GPU with zero disk/decompress/H2D. This is the real fix
    # for the I/O bottleneck (no-cache was ~98s/epoch at 0% GPU util; cached ~8s/epoch incl cold
    # start). cache size = this config's train-frame count. Forces num_workers=0. train data-root
    # is the NVMe stage (fast cold-start read); eval stays on the exfat root (full val/test frames).
    n_cache = _train_frame_count(args.slice, case)
    train_root = args.train_data_root or args.data_root
    # GPU-resident cache is a 12x speedup on Linux (q), but on Windows the cold-start read of
    # zarr chunks memory-maps files that never release page cache — RAM hits 97% used and the
    # training hangs at 0% GPU util after ~1 epoch. Give ops an escape hatch (--no-gpu-cache) to
    # disable the cache; slower but stable. Linux/q side unchanged (still gets the 12x when
    # --no-gpu-cache is omitted).
    #
    # At 256^3 no cache holds ALL frames: 150 frames x 0.13GB (fp16) = 20GB (4ch) / 25GB (scalar
    # 5ch). The GPU cache OOMs the 32GB GPU alongside training; a full CPU cache (19GB for a seed's
    # 146 unique frames) + training (~5GB) + OS (3GB) = 27GB dangerously close to the 30GB physical
    # RAM; and DataLoader workers OOM too (each worker forks its own copy of the frame cache AND
    # prefetches full 256^3 rollout windows -> measured SIGKILL at num_workers=4).
    #
    # HISTORICAL (superseded, kept because the arithmetic still explains the I/O cost): a rollout
    # epoch re-reads its unique train frames ~30x (208 overlapping t_in=20 windows). Uncached and
    # un-overlapped that measured 19.9 min/epoch => ~20 DAYS for the matrix, which is why a full
    # frame cache (~1 min/epoch) was originally planned.
    #
    # That plan is DEAD at the finalized 3-train-seed slice: the cache would need 450 frames =
    # 60.4GB fp16, vs 30GB physical RAM (the ~146-frame/19.6GB budget below assumed ONE train
    # seed). The working substitute is patch-chunk staging + DataLoader workers: a sample reads a
    # 128^3 sub-box instead of a full 256^3 frame, and workers=2 overlaps that read with compute
    # for a measured 6.4 min/epoch at safe RAM headroom (see --no-gpu-cache branch below).
    #
    # --cpu-cache = FULL CPU frame cache (all n_cache train frames, fp16, num_workers 0 so exactly
    # ONE copy). Budget: 146 frames * 0.13GB = 19.6GB steady + ~5GB train + ~3GB OS = 27.6GB, just
    # under 30GB physical RAM. The first-epoch BUILD transiently holds fp32 (0.27GB) + its normalized
    # copy while filling, which can briefly exceed physical RAM — so the scope must ALLOW SWAP (do
    # NOT set MemorySwapMax=0) as a soft backstop for that transient, and MemoryMax should be ~28G.
    # This is the user-chosen path (2026-07-14): add-swap + full-cache, ~1.3 days for the matrix.
    # A partial cache was tried (96 frames) and OOM'd under the old swap-disabled 24G-cap scope.
    if args.cpu_cache:
        # REFUSE if the cache cannot physically fit. It used to be requested, silently not fit, and
        # fall through to reading every window off disk anyway -- so the operator believed they were
        # running the ~1min/epoch cached path while actually running the ~8-16min/epoch I/O path,
        # with the only symptom being "it's slow". Measured on q 2026-07-15: --cpu-cache asked for
        # 450 frames = 56GB against 30GB of RAM, RAM never rose above 3G, and ep000 still took ~8
        # minutes. Same shape as every other bug here: a request that degrades quietly instead of
        # erroring. If you want the I/O path, ask for it by name (--no-gpu-cache).
        _need_gb = n_cache * 4 * (256 ** 3) * 2 / 2 ** 30      # fp16, 4ch, 256^3
        try:
            import psutil as _ps
            _have_gb = _ps.virtual_memory().total / 2 ** 30
        except Exception:
            _have_gb = float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 2 ** 30
        if _need_gb > 0.75 * _have_gb:
            raise SystemExit(
                f"[suite] --cpu-cache needs ~{_need_gb:.1f}GB for {case} ({n_cache} train frames, "
                f"fp16, 4ch, 256^3) but this machine has {_have_gb:.1f}GB of RAM. It would not fit, "
                f"and the cache would silently not build -- you would get the I/O-bound path while "
                f"believing you had the cached one.\n"
                f"  At the finalized 3-train-seed slice (450 frames/config) --cpu-cache is simply "
                f"not viable on a 30GB box; the matrix path is --no-gpu-cache (I/O-bound by design; "
                f"workers=2 was measured to OOM-kill the scope and workers=1 to be SLOWER than 0).")
        cache_args = ["--cache-frames", str(n_cache), "--cache-device", "cpu"]
        nw = "0"   # single process => one cache copy; workers would fork-multiply it and prefetch-OOM
    elif args.no_gpu_cache:
        # No frame cache: every epoch re-reads its windows from the NVMe stage, so this path is
        # I/O-bound. It is nonetheless the MATRIX path, because --cpu-cache no longer fits (see
        # above: 450 frames = 60.4GB vs 30GB physical RAM at the finalized 3-train-seed slice).
        #
        # workers stays 0. DataLoader workers ARE faster in isolation (measured 4.1 min/epoch at
        # workers=2 vs 8.1 at workers=0) but they OOM-kill the run in production: the worker
        # process tree adds a measured +6.9GB on top of the ~20.6GB the training process already
        # holds, i.e. ~27.5GB against the 28G scope cap. That is not a theory -- workers=2 was
        # tried and the scope died with result 'oom-kill' after 8 minutes. workers=1 is pointless
        # (measured 11.5 min/epoch, SLOWER than workers=0 because one worker serializes the read
        # without overlapping enough to pay for the IPC).
        #
        # The lesson from that OOM: a DataLoader benchmarked standalone does NOT predict production
        # memory. Any future attempt to raise this must measure the FULL process tree with the model
        # and CUDA context resident, not a bare DataLoader loop.
        cache_args = ["--cache-frames", "0"]
        nw = str(args.num_workers)
    else:
        cache_args = ["--cache-frames", str(n_cache), "--cache-device", "cuda"]
        nw = "0"   # CUDA cache tensors can't cross a DataLoader worker fork
    cmd = [py, str(HERE / "train_benchmark.py"),
           "--model", model, "--task", task, "--case", case,
           "--slice", args.slice, "--data-root", train_root,
           # train_root may be the NVMe stage (train seeds only); val lives in the full corpus.
           "--val-data-root", args.data_root,
           # Same --seed for every model/task/config: init and shuffle order must not vary across
           # baselines, or the leaderboard partly measures init luck instead of method.
           "--seed", str(args.seed),
           "--epochs", str(args.epochs), "--t-in", "1", "--t-out", str(t_out_train),
           # TRAIN sampling.
           #   rollout        -> sliding window, stride=4 (a window is just 2 frames: input+target)
           #   space tasks    -> per_frame_n=20 equidistant frames per seed (no window at all)
           # These are alternatives, not a pair: rollout uses only the stride, the four single-frame
           # tasks use only per_frame_n.
           #
           # Training stride is --train-frame-stride (default 4 since 2026-07-17). Stride is NOT about
           # sample independence -- independence between two training pairs does not matter at all
           # (overlapping windows are ordinary augmentation; error bars, which is what independence
           # is for, only exist on the eval side). The trade is data volume vs wall-clock. Under the
           # v1 128^3-crop protocol, stride 1/2/4/8 gave 3576/1800/912/456 pairs and stride 4 was
           # the pick; under v2 native (no 8x patch expansion) stride 4 leaves only ~111 pairs, so
           # the default drops to 1 (~447 pairs) to keep the optimizer step budget comparable.
           "--frame-stride", str(args.train_frame_stride), "--per-frame-n", "20",
           "--batch-size", str(phys_b), "--accum-steps", str(accum),
           "--patch", str(_train_patch_for(model, args, models)), "--amp", "--device", args.device,
           "--num-workers", nw,
           *cache_args,
           "--ckpt-dir", str(run_dir)]
    if task == "rollout":
        cmd.append("--residual")   # predict delta + last frame — fixes predict-mean collapse
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.patience > 0:
        cmd += ["--patience", str(args.patience), "--min-delta", str(args.min_delta)]
    cmd += _hp_args(model, models)
    if args.use_wandb:
        cmd.append("--use_wandb")
    if args.max_samples:
        cmd += ["--max-samples", str(args.max_samples)]
    # NOTE: num_workers is hard-set to 0 above — GPU-resident cache tensors (CUDA) can't cross a
    # DataLoader worker fork. Do not append --num-workers here (would override the required 0).
    # Report the cache/worker settings ACTUALLY used. This line used to hardcode "cache={n_cache}@cuda"
    # regardless of which branch ran, so a --no-gpu-cache run advertised a 450-frame CUDA cache it
    # never built -- during an OOM investigation that log sent the reader straight at the wrong cause.
    _cache_desc = f"{cache_args[1]}" + (f"@{cache_args[3]}" if len(cache_args) > 3 else " (none)")
    print(f"  TRAIN {run_dir.name}  (bs={phys_b} accum={accum} eff={phys_b*accum} "
          f"cache={_cache_desc} workers={nw})")
    rc = _run(cmd, run_dir / "train.log")
    if rc != 0:
        _write_status(run_dir, status="train_failed", returncode=rc)
        print(f"  FAILED train (rc={rc}): {run_dir.name}  (see train.log)")
        return "train_failed"

    if task == "rollout" and args.stage2_epochs > 0:
        # Stage-2 of the two-stage recipe (decision tree 2026-07-16, cell 4): fine-tune the
        # stage-1 optimum with the self-corrector objective so the model learns to step from its
        # OWN states (exposure-bias mitigation; see train_benchmark --self-corrector for the
        # measurements). Same data/patch/batch wiring as stage-1; fresh optimizer; no early stop
        # (the definitive C4 run used the full stage-2 budget).
        s2_dir = run_dir / "stage2"
        cmd2 = [py, str(HERE / "train_benchmark.py"),
                "--model", model, "--task", task, "--case", case,
                "--slice", args.slice, "--data-root", train_root,
                "--val-data-root", args.data_root,
                "--seed", str(args.seed),
                "--epochs", str(args.stage2_epochs), "--t-in", "1", "--t-out", str(t_out_train),
                "--frame-stride", str(args.train_frame_stride), "--per-frame-n", "20",
                "--batch-size", str(phys_b), "--accum-steps", str(accum),
                "--patch", str(_train_patch_for(model, args, models)), "--amp", "--device", args.device,
                "--num-workers", nw,
                *cache_args,
                "--ckpt-dir", str(s2_dir),
                "--residual", "--self-corrector",
                "--lr", str(args.stage2_lr),
                "--init-from", str(run_dir / "best.pt")]
        cmd2 += _hp_args(model, models)
        if args.max_samples:
            cmd2 += ["--max-samples", str(args.max_samples)]
        print(f"  STAGE2 {run_dir.name}  (self-corrector {args.stage2_epochs}ep @ lr={args.stage2_lr})")
        rc2 = _run(cmd2, run_dir / "train_stage2.log")
        s2_best = s2_dir / "best.pt"
        if rc2 != 0 or not s2_best.exists():
            _write_status(run_dir, status="stage2_failed", returncode=rc2)
            print(f"  FAILED stage2 (rc={rc2}): {run_dir.name}  (see train_stage2.log)")
            return "stage2_failed"
        # eval reads run_dir/best.pt -> promote the stage-2 model there; keep stage-1 auditable.
        (run_dir / "best.pt").replace(run_dir / "best_stage1.pt")
        import shutil
        shutil.copy2(s2_best, run_dir / "best.pt")

    _write_status(run_dir, status="trained")
    return "trained"


def eval_one(model, task, case, args, models, nonnn=False):
    tag = f"{model}__{task}__{case}"
    run_dir = Path(args.root) / tag
    st = _read_status(run_dir)
    if st.get("status") == "done" and not args.redo:
        return "skipped"
    if st.get("status") == "skipped_oom" and not args.redo:
        print(f"  SKIP eval (oom, permanent): {tag}")
        return "skipped"
    if not nonnn and st.get("status") != "trained":
        print(f"  SKIP eval (not trained): {tag}")
        return "eval_skipped"
    _write_status(run_dir, status="eval_running")

    py = sys.executable
    # rollout eval: 1 frame in, AR unroll 20 steps (1.0 T_L @ dt=0.05); other tasks t_out=1.
    t_out_eval = 20 if task == "rollout" else 1  # needs t_in1+t_out20=21 frames/seed —
    # fits short configs (scalar ~80, most decay); 80@4T_L would exceed short-config lengths (empty
    # eval set) AND be past decorrelation (saturated, non-discriminating metrics). [user-set 2026-07-08]
    is_decay = "decay" in case
    cmd = [py, str(HERE / "eval_benchmark.py"),
           "--model", model, "--task", task, "--case", case,
           *(["--native"] if (args.patch >= 256 and not nonnn) else []),
           "--slice", args.slice, "--data-root", args.data_root,
           # t-in MUST match training (1): the checkpoint's input head is built for it.
           "--t-in", "1", "--t-out", str(t_out_eval), "--device", args.device,
           # Eval sampling: stride 8 (0.4 T_L), per_frame_n 20 -- same order as training.
           #
           # Sampling SPARSELY DOES NOT BUY INDEPENDENCE. Measured ACF on kf4 seed2 and the
           # resulting effective sample size across 3 test seeds:
           #     stride  2 -> 195 samples, rho1=0.92, N_eff 8.9
           #     stride  8 ->  51 samples, rho1=0.69, N_eff 9.3   <- this
           #     stride 64 ->   9 samples, rho1=0.06, N_eff 8.1
           # N_eff is ~9 at EVERY stride: the independent information in a test set is set by
           # (trajectory length x number of seeds), not by how finely it is sliced. The earlier
           # stride=64 threw away 186 samples and bought nothing -- its N_eff was in fact the
           # LOWEST. 51 samples is enough to report mean+/-std and to draw the rollout error
           # curve (9 is not), and matching training's order of magnitude removes a gratuitous
           # train/eval discrepancy. The correlation is handled HONESTLY instead: the paper
           # reports N_eff ~9 and the measured 3.3 T_L decorrelation time rather than pretending
           # 51 samples are independent.
           "--frame-stride", "8", "--per-frame-n", "20",
           "--out", str(run_dir)]
    cmd += _hp_args(model, models)
    if not nonnn:
        # eval the BEST checkpoint (lowest val loss), not the last epoch — matters with early
        # stopping (latest = the patience-expired epoch, best = the actual optimum) and even
        # without it (best.pt is the val-optimal epoch, latest.pt is just epoch 50).
        ckpt = run_dir / "best.pt"
        if not ckpt.exists():
            ckpt = run_dir / "latest.pt"   # fallback (no val set -> no best.pt saved)
        cmd += ["--ckpt", str(ckpt)]
    if task == "rollout" and not nonnn:
        cmd.append("--residual")   # must match how the ckpt was trained (train also --residual)
    if is_decay:
        cmd.append("--decay")  # low-K-safe nRMSE (decay iron rule)
    # eval on a FIXED number of held-out test trajectories (default 32), not the full test set.
    # rollout metrics are per-sample means whose standard error ~1/sqrt(n); N=32 is well within
    # the community norm (PDEBench/APEBench evaluate rollout on fixed 10-50 trajectories) and its
    # SE is tiny vs the between-model gaps a leaderboard reports. It also bounds eval cost — each
    # sample AR-unrolls 20 native-256^3 steps (t_out=20 since 2026-07-15; this comment said 30,
    # the retired protocol) and reads its frames from the exfat corpus. --max-samples (smoke)
    # overrides it when set. NOTE: this DEFAULT TRUNCATES the test set -- a config with 51 nominal
    # test samples is scored on 32. The docs' 51/37/12 are nominal counts, not the leaderboard N.
    n_eval = args.max_samples if args.max_samples else args.eval_samples
    if n_eval:
        cmd += ["--max-samples", str(n_eval)]
    rc = _run(cmd, run_dir / "eval.log")
    if rc != 0:
        _write_status(run_dir, status="eval_failed", returncode=rc)
        print(f"  FAILED eval (rc={rc}): {tag}  (see eval.log)")
        return "eval_failed"
    _write_status(run_dir, status="done")
    print(f"  DONE {tag}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints/bench_q", help="suite output root")
    ap.add_argument("--slice", required=True)
    ap.add_argument("--data-root", default=os.environ.get("TURBGEN_DATA_DIR", ""),
                    help="eval reads from here (full val/test frames, e.g. the exfat corpus)")
    ap.add_argument("--train-data-root", default=os.environ.get("TURBGEN_TRAIN_DIR", ""),
                    help="training reads from here — the NVMe stage of train-seed frames (fast "
                         "cold-start into the GPU cache). Falls back to --data-root if unset.")
    # 25 is the finalized protocol (production_configs_256.json "epochs": 25). The default was 50,
    # which only ever produced 25-epoch runs because the operator passed --epochs 25 by hand: any
    # launch without that override silently trained TWICE the protocol budget and, because
    # CosineAnnealingLR(T_max=epochs) stretches with it, produced a model that is not comparable to
    # the 25-epoch ones. The default must BE the protocol.
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patch", type=int, default=256,
                    help="training crop size passed to train_benchmark. 256 (= native grid) is the "
                         "v2 protocol: full-field training, no crops, no eval stitching. 128 is the "
                         "retired v1 crop protocol (kept for ablation): its 8-tile hard-stitch eval "
                         "injects seam energy that ignites AR self-amplification (measured kf3 "
                         "lr-only 0.8684 vs persistence 0.8407), and a 128^3 crop cannot represent "
                         "k<2 modes of the box-scale forcing.")
    ap.add_argument("--train-frame-stride", type=int, default=4,
                    help="frame stride for TRAINING sampling. DEFAULT 4 since 2026-07-17 (deadline "
                         "protocol, user decree): stride=1 is permanently retired for NEW training; "
                         "existing stride-1 rows (kf3/relam70/relam50) are kept as official and "
                         "pooled with stride-4 rows (adjacent frames ~1 tau_eta apart are highly "
                         "correlated; the kf3/relam70 {1,4,8} bridge ablation quantifies the delta).")
    ap.add_argument("--lr", type=float, default=None,
                    help="stage-1 learning rate passed through to train_benchmark (None = its "
                         "default). The 2026-07-15 diagnosis measured lr=1e-4 collapsing every "
                         "rollout model into persistence (f(x) ~ 1%% of the true delta) within the "
                         "production step budget; the two-stage recipe runs stage-1 at 1e-3.")
    ap.add_argument("--stage2-epochs", type=int, default=0,
                    help="rollout only: epochs of --self-corrector fine-tuning appended after "
                         "stage-1 (0 = off, default = protocol unchanged). The decision tree of "
                         "2026-07-16 (C1 FAIL x C4b FAIL -> cell 4) fixed the production recipe to "
                         "stage-1 lr=1e-3 x 25ep + stage-2 self-corrector 5ep@1e-3: on relam70 it "
                         "is the only recipe that beat persistence on the official nRMSE_mean "
                         "(0.7581 vs 0.8217), at a documented -7.6%% s0 cost (intrinsic to the "
                         "corrector objective; intensity-invariant). After stage-2 the run's "
                         "best.pt is the stage-2 model (stage-1 kept as best_stage1.pt), so eval "
                         "scores the full recipe. Resume note: a run already marked 'trained' "
                         "skips training entirely, so flipping this flag on later does NOT "
                         "retrofit stage-2 onto old runs -- use --redo for that.")
    ap.add_argument("--stage2-lr", type=float, default=1e-3,
                    help="learning rate for the stage-2 self-corrector fine-tune (default 1e-3, "
                         "the C4 definitive-run value).")
    ap.add_argument("--seed", type=int, default=0,
                    help="ML RNG seed (weight init + shuffle order), passed to every run. Distinct "
                         "from the corpus/DNS seeds in the slice.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", nargs="*", default=Q_CONFIGS, help="subset of configs")
    ap.add_argument("--tasks", nargs="*", default=list(REQUIRED), help="subset of tasks")
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="smoke: cap BOTH train and eval samples (overrides --eval-samples).")
    ap.add_argument("--eval-samples", type=int, default=51,
                    help="held-out test samples per eval = the leaderboard N. DEFAULT 51 = the "
                         "full test split (the largest nominal count across configs), i.e. NO "
                         "truncation: every config is scored on all the samples its slice gives "
                         "it (51 / 37 / 12 depending on the case). "
                         "This default was 32, which SILENTLY TRUNCATED the 51- and 37-sample "
                         "configs -- the docs' test counts then described a run nobody did. The "
                         "eval saving was not worth publishing a number we had not computed. "
                         "N_eff is ~9 regardless (independence comes from trajectory length x "
                         "seeds, not from slicing finer), so 51 buys curve resolution and honest "
                         "bookkeeping, not extra independence -- report N_eff, not N. "
                         "ALL THREE MACHINES MUST USE THE SAME VALUE or the leaderboard is not "
                         "comparable across configs.")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. Training hard-sets 0 regardless (GPU-resident cache "
                         "CUDA tensors can't cross a worker fork); kept for compatibility/eval.")
    ap.add_argument("--redo", action="store_true", help="force re-run all (ignore done status)")
    ap.add_argument("--dry-run", action="store_true", help="print the matrix + resume state, no runs")
    ap.add_argument("--models", nargs="*", default=None,
                    help="allow-list of NN models to include (filters REQUIRED). Ex: "
                         "--models fno3d transolver ufno   (skip tfno on Windows where "
                         "expandable_segments is unsupported). None = all REQUIRED models. "
                         "NOTE: dpot was dropped from REQUIRED in 2026-07; it stays importable "
                         "but naming it here runs a model that is not in the benchmark.")
    ap.add_argument("--no-nonnn", action="store_true",
                    help="skip non-NN eval baselines (persistence/spectral_interp/identity/smagorinsky). "
                         "Useful for a NN-only sweep; non-NN can be run separately with the same script.")
    ap.add_argument("--patience", type=int, default=8,
                    help="Early-stopping patience passed to train_benchmark: stop if the monitored "
                         "loss doesn't improve >min-delta for N epochs. Default 8. 0 = disabled.")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="Relative improvement threshold for early stopping (default 1e-4 = 0.01%%). "
                         "Early stopping is a SAFETY NET (dead/diverged run); the cosine schedule "
                         "over --epochs is the normal stop, so this stays small on purpose.")
    ap.add_argument("--no-gpu-cache", action="store_true",
                    help="Disable GPU-resident frame cache (--cache-frames 0). Windows only: "
                         "the cold-start read of 100 zarr chunks memory-maps files that page cache "
                         "never releases -> RAM 97%%, GPU stalls at 0%% util. Slower per-epoch but "
                         "STABLE — use on Windows. Linux (q) keeps cache = 12x speedup.")
    ap.add_argument("--cpu-cache", action="store_true",
                    help="FULL CPU frame cache (all train frames, fp16, num_workers 0). The 256^3 "
                         "GPU cache OOMs the 32GB VRAM; this keeps the ~146 unique train frames in "
                         "host RAM (~19.6GB) instead, cutting per-epoch reads from ~878GB to ~0 "
                         "(~16.5min/epoch -> ~1min). REQUIRES a scope that allows swap (do NOT set "
                         "MemorySwapMax=0) with MemoryMax~28G, as the first-epoch build transiently "
                         "spikes near physical RAM. Takes precedence over --no-gpu-cache.")
    args = ap.parse_args()

    models = _load_models()
    Path(args.root).mkdir(parents=True, exist_ok=True)

    # build the run list: (model, task, case, is_nonnn), applying optional filters
    model_allow = set(args.models) if args.models else None
    runs = []
    for case in args.configs:
        for task in args.tasks:
            required_here = list(REQUIRED.get(task, []))
            # Appendix models (e.g. dpot) live in MODEL_REGISTRY but NOT in REQUIRED -- the header
            # docstring says they "run ONLY via explicit --models". Honor that: a --models entry
            # that is a real registered model but absent from REQUIRED[task] is an explicit
            # appendix request, so schedule it (2026-07-24, final-48h dpot 5th-baseline column).
            if model_allow is not None:
                for m in model_allow:
                    # `models` = production_configs "models" dict (has a "dpot" key). An explicit
                    # --models entry that has a production config but is absent from REQUIRED[task]
                    # is a valid appendix request.
                    if m not in required_here and m in models:
                        required_here.append(m)
            for model in required_here:
                if model_allow is not None and model not in model_allow:
                    continue
                runs.append((model, task, case, False))
            if not args.no_nonnn:
                for model in NONNN.get(task, []):
                    runs.append((model, task, case, True))

    # resume snapshot
    done = sum(1 for m, t, c, _ in runs
               if _read_status(Path(args.root) / f"{m}__{t}__{c}").get("status") == "done")
    print(f"[suite] {len(runs)} runs total ({len(args.configs)} configs x {len(args.tasks)} tasks), "
          f"{done} already done, {len(runs)-done} to go. root={args.root}")
    if args.dry_run:
        for m, t, c, nn in runs:
            s = _read_status(Path(args.root) / f"{m}__{t}__{c}").get("status", "queued")
            print(f"    [{s:>13}] {'(nonNN)' if nn else '       '} {m}__{t}__{c}")
        return

    summary = {"done": 0, "skipped": 0, "failed": 0}
    for model, task, case, nn in runs:
        if not nn:
            r = train_one(model, task, case, args, models)
            if r == "train_failed":
                summary["failed"] += 1
                continue  # don't eval a failed train
            if r == "skipped":
                summary["skipped"] += 1
                continue
        r = eval_one(model, task, case, args, models, nonnn=nn)
        if r == "done":
            summary["done"] += 1
        elif "failed" in r:
            summary["failed"] += 1
        elif "skipped" in r:
            summary["skipped"] += 1

    print(f"\n[suite] finished: {summary}")


if __name__ == "__main__":
    main()
