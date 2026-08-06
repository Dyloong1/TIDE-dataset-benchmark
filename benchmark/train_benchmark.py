"""train_benchmark.py — train ONE baseline on ONE task/config slice of the turbgen corpus.

Multi-task: --task {rollout,superres,recon,pressure,sgs}. Reads TurbgenZarrDataset (zarr +
benchmark_slice.json + frozen normalizer), instantiates a curated baseline by name, trains
with MSE (or the model's get_training_loss for diffusion/refiner/flow-matching models —
gated on hasattr, so flow_refiner uses flow-matching, not bare MSE), cosine LR, optional AMP
(--amp), optional wandb scalar logging (--use_wandb), best.pt + latest.pt checkpointing.
(EMA is NOT built into this CLI; attach a model-side EMA if a generative baseline needs it.)
Output shapes (T_in/T_out, channels) come from the dataset so the same loop serves every task.

  python train_benchmark.py --model fno3d --task rollout --case ou_relam70_256_fp64 \
      --slice $TURBGEN_DATA_DIR/corpus/benchmark_slice.json --epochs 25 \
      --ckpt-dir checkpoints/bench_fno3d_rollout_relam70

Heavy runs wait for free GPU (kf4 production); use --max-samples / tiny configs for smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Repo rule: import solver._env BEFORE torch. Windows ships two OpenMP runtimes (torch + MKL) and
# aborts with "OMP: Error #15" without KMP_DUPLICATE_LIB_OK, which _env sets at import time. This
# runs on CYB-t (Windows), so relying on a launch script to export the env var is not good enough.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import solver._env  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.baselines import get_model, MODEL_REGISTRY  # noqa: E402
from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset  # noqa: E402

# channels per task: rollout/superres/recon keep full corpus channels; pressure 3→1; sgs 4→6
DIFFUSION_MODELS = {"pde_refiner", "acdm"}


def _predict(model, x, residual):
    """Model forward with optional RESIDUAL (delta) prediction for rollout.

    Without residual the model must regress the absolute next frame u(t+1). Because consecutive
    turbulence frames are highly correlated (frame-to-frame change << field amplitude), the MSE
    landscape has a trivial 'predict the mean (≈0 in normalized space)' basin whose loss ≈ var(y);
    models collapse into it (measured: predict-zero MSE 0.017 ≈ train loss, while persistence MSE
    is 0.005 — 3x better but never found). Residual makes the model predict the DELTA and adds back
    the last input frame: pred = x[:, -1] + f(x). Now persistence is 'output 0 delta' (the natural
    baseline) and the model only has to learn the small correction — the standard fix for temporal
    NO rollout (APEBench/PDEBench). Applied only when out shape matches the last input frame
    (rollout: same C, T_out frames); other tasks are unaffected.
    """
    out = model(x)                      # (B, T_out, C, ...)
    if not residual:
        return out
    last = x[:, -1:]                    # (B,1,C_in,...) last input frame
    if out.shape[1:] == last.shape[1:] or (out.shape[2] == last.shape[2] and out.shape[1] >= 1):
        # broadcast the last frame across T_out output frames (C must match)
        if out.shape[2] == last.shape[2]:
            return out + last[:, :1]
    return out                          # shapes incompatible (channel-changing task) -> no residual


import json as _json
_PROD_CFG = Path(__file__).resolve().parent / "Model" / "configs" / "production_configs_256.json"


def _arch_keys_from_json(model):
    """Load the model's architecture hyperparams from production_configs_256.json so the
    transformer/generative models (dpot/transolver/acdm/pde_refiner) actually run at their
    DOCUMENTED capacity instead of falling back to code defaults. Skips _meta fields. Returns
    {} if the file/model is absent (models then use their getattr defaults, as before). This is
    what wires production_configs into the harness — without it the JSON was decorative and e.g.
    dpot ran at 3M (default dim=256->wait, dims) instead of its 27.5M documented config."""
    try:
        models = _json.loads(_PROD_CFG.read_text(encoding="utf-8"))["models"]
    except Exception:
        return {}
    return {k: v for k, v in (models.get(model) or {}).items() if not k.startswith("_")}


def build_config(args, in_ch, out_ch, t_in, t_out, model=None, grid_n=None):
    """A duck-typed config the models read via getattr (BaselineConfig-compatible). Architecture
    hyperparams come from production_configs_256.json (via _arch_keys_from_json) so every model
    runs at its documented capacity; CLI --width/--modes/etc. still override the FNO-family keys
    (back-compat). TOTAL_Z_LAYERS is set from the field grid so patch-transformers size correctly."""
    inds = ['u', 'v', 'w', 'p'][:in_ch] if in_ch <= 4 else ['c%d' % i for i in range(in_ch)]
    base = dict(
        INPUT_TIMESTEPS=t_in, OUTPUT_TIMESTEPS=t_out,
        INDICATORS=inds, IN_CHANNELS=in_ch, OUT_CHANNELS=out_ch,
        FNO_WIDTH=args.width, FNO_LAYERS=args.layers,
        FNO_MODES_Z=args.modes, FNO_MODES_H=args.modes, FNO_MODES_W=args.modes,
        TFNO_RANK=args.tfno_rank, DEEPONET_P=args.deeponet_p,
    )
    if grid_n:
        base["TOTAL_Z_LAYERS"] = int(grid_n)
    # Architecture keys from production_configs are the documented capacity for every model whose
    # key the CLI does not own. Only the FNO-family keys above are seeded from the CLI (the suite's
    # _hp_args emits --width/--layers/--modes/--tfno-rank/--deeponet-p for them); every other key
    # (UFNO_WIDTH, the transolver/dpot/deeponet/generative keys) MUST come from the JSON, so it is
    # NOT pre-seeded here — a pre-seeded key would win over setdefault and silently pin the model to
    # the argparse default instead of its documented capacity.
    for k, v in _arch_keys_from_json(model).items():
        base.setdefault(k, v)
    # per-model fallbacks when the JSON lacks the key (keeps a bare CLI run working).
    base.setdefault("UFNO_WIDTH", args.width)
    base.setdefault("DEEPONET_WIDTH", args.width)
    return SimpleNamespace(**base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--task", required=True, choices=("rollout", "superres", "recon", "pressure", "sgs"))
    ap.add_argument("--case", required=True)
    ap.add_argument("--slice", required=True, help="benchmark_slice.json")
    ap.add_argument("--epochs", type=int, default=25,
                    help="Finalized protocol: 25, and the run is EXPECTED to use all of them -- "
                         "CosineAnnealingLR(T_max=epochs) anneals fully within the budget, so a "
                         "full 25-epoch run is the normal ending and early stopping is only a "
                         "safety net. The default was 50, which silently doubled the budget (and, "
                         "because T_max follows it, produced a model not comparable to the "
                         "25-epoch ones) for anyone who did not pass --epochs explicitly.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="gradient accumulation: opt.step() every N micro-batches. "
                         "effective_batch = batch_size * accum_steps. Production pins "
                         "effective_batch=16 (compute-budget comparability across baselines).")
    ap.add_argument("--val-batch-size", type=int, default=1,
                    help="Batch for the val loop. val is patched by default (--val-patch), so this "
                         "is no longer VRAM-critical; TEST eval still runs native 256^3 where a "
                         "single sample can peak ~28GB, hence the conservative default of 1.")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Peak LR for AdamW, annealed to ~0 by CosineAnnealingLR(T_max=--epochs). "
                         "The SAME lr is used for every baseline on purpose: per-model LR tuning "
                         "would add a tuning-budget confound to the leaderboard, and the protocol's "
                         "comparability contract is one fixed recipe for all models. 1e-4 is the "
                         "standard operating point for this operator family and is verified to train "
                         "here (fno3d rollout: 2.8e-3 -> 7.5e-4, a 73%% drop in 16 epochs, smooth, "
                         "no divergence).")
    ap.add_argument("--t-in", type=int, default=1)    # NS is Markovian: u(t) is a complete state
    ap.add_argument("--t-out", type=int, default=1)   # single-step training (protocol §4); eval_benchmark passes t-out=30 (1.5 T_L AR) explicitly
    ap.add_argument("--frame-stride", type=int, default=4,
                    help="rollout sliding-window step (protocol §109: stride=5 => ~26 pairs/seed, "
                         "removes adjacent-window redundancy). Space tasks ignore it (use --per-frame-n).")
    ap.add_argument("--per-frame-n", type=int, default=20,
                    help="space tasks (superres/recon/pressure/sgs): equidistant frames per seed "
                         "(protocol §110). Deterministic count regardless of trajectory length so "
                         "cross-config sample count + training cost do not drift (relam90 250 frames "
                         "still yields 20, same as kf4 150).")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--modes", type=int, default=16)
    ap.add_argument("--tfno-rank", type=int, default=8)
    ap.add_argument("--deeponet-p", type=int, default=128)
    ap.add_argument("--ckpt-dir", default="checkpoints/bench_run")
    ap.add_argument("--data-root", default=os.environ.get("TURBGEN_DATA_DIR", ""))
    ap.add_argument("--val-patch", dest="val_patch", action="store_true", default=True,
                    help="Validate on 128^3 patches like training (default). A native-256^3 val "
                         "sample costs ~5.4GB host RAM on top of the training process and OOM-kills "
                         "a 28G cgroup. val only drives early stopping / best.pt and never enters "
                         "the paper; TEST eval stays native 256^3 regardless.")
    ap.add_argument("--no-val-patch", dest="val_patch", action="store_false",
                    help="Validate on the native 256^3 field. Needs headroom well above 28G.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for weight init + train shuffle. Pinning this is what makes a "
                         "run reproducible AND makes the baselines comparable: without it every "
                         "model draws a different init and a different gradient-step ordering, so "
                         "part of any leaderboard gap is luck. Distinct from the DNS/corpus seeds "
                         "in the slice (those select trajectories, not training randomness).")
    ap.add_argument("--val-data-root", default="",
                    help="Corpus root for the VAL split. Defaults to --data-root. Needed because "
                         "--data-root may point at an NVMe stage that holds ONLY the train seeds "
                         "(stage_train_to_nvme.py stages train data); the val seeds live in the "
                         "full corpus. Without this, val silently yields 0 samples and early "
                         "stopping/best.pt fall back to TRAIN loss, which cannot see overfitting.")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patch", type=int, default=128,
                    help="train on (patch)^3 crops of the 256^3 field (256^3 full-field OOMs); "
                         "0/None = full field. val/test always use the full native field.")
    ap.add_argument("--amp", action="store_true", help="mixed-precision (bf16 autocast, no GradScaler — complex-weight safe)")
    ap.add_argument("--cache-frames", type=int, default=0,
                    help="LRU-cache this many decompressed train frames (train dataset only). "
                         "The bottleneck is re-loading frames shared by overlapping rollout windows "
                         "each epoch (~190ms/frame: zstd decompress + normalize). 0 = disabled.")
    ap.add_argument("--cache-device", default="cpu",
                    help="where the frame cache lives: 'cuda' keeps decompressed fp16 frames "
                         "RESIDENT IN VRAM (0.13GB/frame; 100 frames=13GB fits the 5090 alongside "
                         "128^3-patch training) — eliminates disk+decompress+H2D after epoch 0. "
                         "Requires --num-workers 0 (CUDA tensors can't cross DataLoader workers).")
    ap.add_argument("--use_wandb", action="store_true",
                    help="log scalar train/val loss + lr to wandb (lightweight: no artifacts/media)")
    ap.add_argument("--residual", action="store_true",
                    help="ROLLOUT: predict the delta and add back the last input frame "
                         "(pred = x[:,-1] + f(x)). Fixes the 'predict-mean/zero collapse' where "
                         "absolute-frame regression falls into the trivial basin. Standard for "
                         "temporal NO rollout; use for rollout, harmless-but-pointless elsewhere.")
    ap.add_argument("--self-corrector", action="store_true",
                    help="ROLLOUT+--residual only (t_in=1): one no-grad warmup builds the model's "
                         "OWN next-state estimate x1 = x + f(x); the loss then trains f(x1) against "
                         "the SAME true frame u(t+1) -- i.e. teach the model to step back onto the "
                         "truth trajectory from its own slightly-wrong states (a contraction "
                         "property). NOTE this is NOT textbook push-forward (that would target "
                         "u(t+2) and needs t_out=2); it is the variant the CYB-3 probe actually "
                         "validated: same fine-tune budget from the diseased ckpt gave s19 "
                         "enstrophy 3.87 vs 7.44 (standard) vs 21.5 (none), with |f| AR "
                         "amplification eliminated (1.0x vs 6.7x). Known risk: over-damping |f| "
                         "toward collapse -- always report |f| alongside. Off by default; enabling "
                         "it changes the training protocol.")
    ap.add_argument("--init-from", default=None,
                    help="path to a checkpoint whose model_state_dict initialises the model before "
                         "training (two-stage recipes: e.g. single-step pretrain -> --self-corrector "
                         "fine-tune). Architecture must match; optimizer/scheduler start fresh.")
    ap.add_argument("--patience", type=int, default=0,
                    help="Early-stopping patience: stop if the monitored loss doesn't improve for "
                         "N consecutive epochs (0 = disabled, train full --epochs). Monitors val "
                         "loss when a val set exists, else train loss. Uses --min-delta threshold.")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="Minimum RELATIVE improvement counted as progress for early stopping "
                         "(loss_new < best * (1 - min_delta)). Default 1e-4 = 0.01%. Early stopping "
                         "is a SAFETY NET here (a dead or diverged run), not the normal stop: the "
                         "cosine schedule over --epochs is what ends training, and the loss keeps "
                         "improving until the LR anneals, so a larger threshold would cut the "
                         "schedule short.")
    ap.add_argument("--min-epochs", type=int, default=10,
                    help="Do not early-stop before this epoch (guards against premature stops on "
                         "noisy early-epoch loss). Default 10.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Linear LR ramp over the first N OPTIMIZER steps (0 = off, protocol "
                         "unchanged). Probe flag for the transformer-family basin stall: "
                         "Adam+transformer cold-started at peak LR is the classic ViT stall mode, "
                         "while the FNO family trains fine without warmup. Multiplies the cosine "
                         "schedule's LR; hands control back to it after the ramp.")
    # task definition parameters — explicit so the protocol is auditable from the command line
    # rather than buried in dataset defaults (train and eval must pass the SAME values).
    ap.add_argument("--sr-factor", type=int, default=4,
                    help="P2 super-resolution downsample factor (default 4 = x4 SR).")
    ap.add_argument("--recon-keep-frac", type=float, default=0.05,
                    help="P3 sparse-reconstruction kept-pixel fraction (default 0.05 = 95%% of the "
                         "field is masked out). Single operating point; the mask is deterministic "
                         "per (seed,frame,patch) so every model sees identical observations.")
    ap.add_argument("--sgs-sigma-frac", type=float, default=0.04,
                    help="P5 SGS Gaussian filter width as a fraction of the box (default 0.04).")
    args = ap.parse_args()

    dev = torch.device(args.device)
    common = dict(case=args.case, slice_manifest=args.slice, task=args.task,
                  data_root=args.data_root, max_samples=args.max_samples,
                  frame_stride=args.frame_stride, per_frame_n=args.per_frame_n,
                  sr_factor=args.sr_factor, recon_keep_frac=args.recon_keep_frac,
                  sgs_sigma_frac=args.sgs_sigma_frac)
    # train crops 128^3 patches (256^3 full-field training OOMs); val sees full native field
    # (patch_size only takes effect on split=='train' inside the dataset).
    tr = TurbgenZarrDataset(split="train", t_in=args.t_in, t_out=args.t_out,
                            patch_size=args.patch, cache_frames=args.cache_frames,
                            cache_device=args.cache_device, **common)
    # val reads from the FULL corpus, not the train stage: --data-root may be an NVMe stage that
    # holds only the train seeds, in which case the val seeds are absent there and val would come
    # back empty without ever raising.
    #
    # val is PATCHED like train (--val-patch, default on). A native-256^3 val sample is 20 frames x
    # 4ch x 256^3 fp32 = 5.4GB of host RAM and ~1.4GB of forward overhead, which lands on top of the
    # ~20.6GB the training process already holds -> ~27.4GB against a 28G cgroup cap. That is not
    # hypothetical: it OOM-killed the suite twice, both times at the 8-12 min mark, i.e. exactly when
    # epoch 0 ended and validation began. (Earlier runs "survived" only because val was silently
    # EMPTY -- the bug fixed just above.)
    #
    # This costs nothing scientifically: val exists solely to drive early stopping and pick best.pt,
    # and never appears in the paper. Patched val is also the SAME distribution the model trains on,
    # while TEST eval stays native 256^3 (see eval_benchmark.py), so no published number is affected.
    val_common = dict(common, data_root=args.val_data_root or args.data_root)
    # NOTE: the dataset gates patching on `split == "train" or eval_patch` (see its __init__), so
    # passing patch_size alone on a val split is silently ignored -- eval_patch is the lever.
    va = TurbgenZarrDataset(split="val", t_in=args.t_in, t_out=args.t_out,
                            patch_size=(args.patch if args.val_patch else None),
                            eval_patch=bool(args.val_patch),
                            **val_common)
    if len(tr) == 0:
        raise SystemExit(f"[train] empty train set for {args.case}/{args.task} — check slice manifest")
    # An empty val set is never intentional: the slice always assigns a val seed, so 0 samples means
    # the data is missing from this root (staged train-only) or the window filter dropped everything.
    # Fail loudly instead of silently degrading early stopping + best.pt selection to train loss,
    # which cannot detect overfitting -- the entire reason val exists.
    if len(va) == 0:
        val_seeds = (json.loads(Path(args.slice).read_text(encoding="utf-8"))
                     .get("per_config", {}).get(args.case, {}).get("val", {}).get("seeds", []))
        raise SystemExit(
            f"[train] empty VAL set for {args.case}/{args.task}: slice assigns val seeds "
            f"{val_seeds} but 0 samples were built from {val_common['data_root']!r}. If that is an "
            f"NVMe train stage, pass --val-data-root <full corpus root>.")

    # Seed BEFORE the model is built: get_model draws the weight init from the global RNG, so this
    # line is what makes init reproducible and identical across baselines. Without it, two runs of
    # the same config produced different weights (verified), meaning no published number could be
    # reproduced and part of any model-vs-model gap was init luck rather than method.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sample = tr[0]
    in_ch = sample["input_dense"].shape[1]
    out_ch = sample["output_dense"].shape[1]
    t_in = sample["input_dense"].shape[0]
    t_out = sample["output_dense"].shape[0]
    grid_n = sample["input_dense"].shape[-1]   # 128 (train patch); patch-transformers size to this
    cfg = build_config(args, in_ch, out_ch, t_in, t_out, model=args.model, grid_n=grid_n)
    model = get_model(args.model, cfg).to(dev)
    if args.init_from:
        _init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(_init["model_state_dict"])
        print(f"[train] initialised from {args.init_from} (epoch {_init.get('epoch')})")
    if args.self_corrector and (not args.residual or t_in != 1):
        raise SystemExit("[train] --self-corrector requires --residual and t_in=1 (the warmup step "
                         "x1 = x + f(x) only type-checks in that configuration).")

    # fail-fast task/model channel compatibility check (BEFORE burning GPU-hours on training).
    # Only fno3d/tfno/deeponet/dpot honor OUT_CHANNELS != IN_CHANNELS; the rest
    # hardcode out == in. On out!=in tasks (pressure 3->1, sgs 4->6) an incompatible model would
    # otherwise SILENTLY broadcast in mse_loss (3ch pred vs 1ch target -> UserWarning only) and
    # train to a meaningless loss that still lands a leaderboard number. Per design, pressure/sgs
    # are assigned only to the 4 channel-flexible models — enforce that here instead of trusting it.
    with torch.no_grad():
        # sample["input_dense"] is (T_in, C, Z, H, W); add a batch dim -> (1, T_in, C, Z, H, W)
        probe = sample["input_dense"].unsqueeze(0).to(dev)
        try:
            pred_ch = model(probe).shape[-4]
        except Exception as e:
            raise SystemExit(f"[train] model '{args.model}' failed forward on task '{args.task}' "
                             f"(in_ch={in_ch}, out_ch={out_ch}): {e}")
    if pred_ch != out_ch:
        raise SystemExit(
            f"[train] INCOMPATIBLE model/task: '{args.model}' outputs {pred_ch} channels but "
            f"task '{args.task}' needs {out_ch} (in={in_ch}). This model hardcodes out==in and "
            f"cannot do channel-changing tasks (pressure 3->1, sgs 4->6). Per design, pressure/"
            f"sgs use only fno3d/tfno/deeponet. Refusing to train a broadcast-garbage model.")

    # Any model exposing get_training_loss uses it (its own diffusion / flow-matching loss),
    # else plain MSE. This includes flow_refiner (flow-matching) — it must NOT be trained with
    # bare MSE, so gate on the method's presence, not a hardcoded name set (DIFFUSION_MODELS
    # kept for reference/labeling).
    is_diff = hasattr(model, "get_training_loss")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # AMP = bf16 autocast, NO GradScaler. bf16 has fp32's exponent range so loss-scaling
    # is unnecessary (Lightning even forbids a scaler with bf16). Critically, GradScaler.unscale_
    # has no complex-dtype CUDA kernel (pytorch #58228, open since 2021), so it crashes on the
    # ComplexFloat spectral weights of FNO/UFNO/TFNO. bf16-no-scaler sidesteps that entirely and
    # matches the production plan's stated bf16-AMP scheme. (fp16 AMP is intentionally not offered.)
    amp_on = args.amp and dev.type == "cuda"
    # I/O is the bottleneck: each 256^3 frame is one zstd-compressed zarr chunk, and training
    # re-loads the same ~100 train frames every epoch (~190ms/frame decompress+normalize; GPU
    # sits at 0% util waiting). Best fix is --cache-device cuda (frames resident in VRAM). When
    # the cache is on GPU the batch tensors are already CUDA, so pin_memory must be OFF (can't pin
    # CUDA tensors) and workers must be 0 (CUDA tensors can't cross a DataLoader fork).
    gpu_cache = (args.cache_frames > 0 and args.cache_device == "cuda")
    nw = 0 if gpu_cache else args.num_workers   # GPU-cached CUDA tensors can't cross a worker fork
    if gpu_cache and args.num_workers > 0:
        print(f"[train] --cache-device cuda forces num_workers=0 (was {args.num_workers})")
    dl_kw = dict(num_workers=nw, pin_memory=(dev.type == "cuda" and not gpu_cache))
    if nw > 0:
        # prefetch_factor=2 (not higher): each prefetched sample holds 256^3 frames in RAM, and
        # workers*prefetch*frames must stay within the box's ~20GB free — 6 workers x 2 is safe.
        dl_kw.update(persistent_workers=True, prefetch_factor=2)
    # Explicit generator: DataLoader's shuffle otherwise draws from the global RNG, whose state at
    # this point depends on how many init draws the model happened to make -- so the step ordering
    # would differ per architecture even with a fixed --seed. A dedicated generator makes the
    # ordering a function of --seed alone, identical for every baseline (comparability contract).
    _shuffle_gen = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    generator=_shuffle_gen, **dl_kw)
    vl = DataLoader(va, batch_size=args.val_batch_size, shuffle=False, **dl_kw) if len(va) else None

    wb = None
    if args.use_wandb:
        import wandb  # lightweight: scalar metrics only, no artifacts/media/code (CLAUDE.md rule)
        wb = wandb.init(project="turbgen-benchmark",
                        name=f"{args.model}_{args.case}_{args.task}",
                        config=vars(args), save_code=False,
                        settings=wandb.Settings(console="off", _save_requirements=False),
                        dir=str(Path(args.ckpt_dir)))

    ckpt_dir = Path(args.ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    epochs_since_improve = 0    # early-stopping counter
    print(f"[train] {args.model} task={args.task} case={args.case} "
          f"in=({t_in},{in_ch}) out=({t_out},{out_ch}) train={len(tr)} val={len(va)} dev={dev}")
    if args.patience > 0:
        print(f"[train] early stopping: patience={args.patience} epochs, min_delta={args.min_delta} (relative)")

    accum = max(1, args.accum_steps)
    # guard: if the loader has fewer micro-batches than one accum-group (tiny --max-samples
    # smoke runs), n_full would be 0 and the epoch would silently take ZERO optimizer steps.
    # Shrink accum to the available count so we still train (and effective_batch shrinks with
    # it) instead of no-op-ing. Production (single seed ~95-145 windows >> 16) never hits this.
    if len(tl) < accum:
        print(f"[train] WARNING: only {len(tl)} micro-batches < accum={accum}; "
              f"shrinking accum to {max(1, len(tl))} (effective_batch reduced for this run)")
        accum = max(1, len(tl))
    # only full accum-groups are stepped, so every optimizer step sees EXACTLY
    # effective_batch = batch_size * accum samples. A trailing partial group is dropped
    # (mirrors DataLoader drop_last=True): flushing it would step at a smaller effective
    # batch (each micro-loss is scaled by 1/accum, so a k<accum tail runs at k/accum scale),
    # which silently breaks the "effective_batch=16 for every baseline" compute-comparability
    # claim. With shuffle=True the dropped tail is a different handful of samples each epoch.
    n_full = (len(tl) // accum) * accum
    # --warmup-steps: linear LR ramp over the first N OPTIMIZER steps, multiplied onto whatever
    # the per-epoch cosine schedule sets. Exists because Adam+transformer cold-started at peak LR
    # is the classic stall (noisy second-moment estimates early); the FNO family trains fine
    # without it, which is why 0 stays the default (protocol byte-identical when unused).
    opt_step_count = 0
    for ep in range(args.epochs):
        model.train(); tot = 0.0; n = 0
        opt.zero_grad()
        # grad accumulation: sum grads over `accum` micro-batches, step once. Scale each
        # micro-loss by 1/accum so the accumulated grad == grad of the true (batch*accum)-sized
        # mean-reduced batch. Step + zero on accum boundaries; samples past the last full
        # group (i >= n_full) are skipped so every step is a full effective batch.
        for i, b in enumerate(tl):
            if i >= n_full:  # drop trailing partial group (keeps effective_batch exact)
                break
            x = b["input_dense"].to(dev); y = b["output_dense"].to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp_on):
                if args.self_corrector and not is_diff:
                    # exposure-bias mitigation: warm up one step WITHOUT gradient so the input the
                    # loss sees is the model's own predicted state, not a ground-truth frame. The
                    # target stays the true next frame, so the model learns to step correctly FROM
                    # its own (slightly wrong) states -- the situation AR rollout actually puts it in.
                    with torch.no_grad():
                        x_self = _predict(model, x, args.residual)
                    loss = nn.functional.mse_loss(_predict(model, x_self, args.residual), y)
                else:
                    loss = (model.get_training_loss(x, y) if is_diff
                            else nn.functional.mse_loss(_predict(model, x, args.residual), y))
            (loss / accum).backward()  # no GradScaler: bf16 needs no loss-scaling (see amp_on note above)
            tot += loss.item(); n += 1
            if (i + 1) % accum == 0:
                if args.warmup_steps > 0 and opt_step_count < args.warmup_steps:
                    ramp = (opt_step_count + 1) / args.warmup_steps
                    cos_lr = sched.get_last_lr()[0]
                    for g in opt.param_groups:
                        g["lr"] = cos_lr * ramp
                elif args.warmup_steps > 0 and opt_step_count == args.warmup_steps:
                    for g in opt.param_groups:   # hand LR control back to the cosine schedule
                        g["lr"] = sched.get_last_lr()[0]
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad()
                opt_step_count += 1
        sched.step()
        tr_loss = tot / max(n, 1)

        vmsg = ""; vloss = None
        if vl is not None:
            if dev.type == "cuda":
                torch.cuda.empty_cache()   # release train activations pool before native-256 val (32GB VRAM safety)
            model.eval(); vt = 0.0; vn = 0
            with torch.no_grad():
                for b in vl:
                    x = b["input_dense"].to(dev); y = b["output_dense"].to(dev)
                    vt += nn.functional.mse_loss(_predict(model, x, args.residual), y).item(); vn += 1
            vloss = vt / max(vn, 1); vmsg = f" val={vloss:.6f}"
        # Early-stopping / best-ckpt monitor: use val loss when a val set exists, else fall back to
        # TRAIN loss. Staging only stages train-seed frames onto the NVMe root, so when eval reads
        # the same single --data-root the val seed is absent and len(va)==0 -> vl is None. Without
        # a train-loss fallback the patience guard below (previously gated on `vl is not None`) never
        # fired, so every run burned all --epochs regardless of convergence. The criterion is the same
        # for every model on a given config, so comparability is preserved either way.
        monitor = vloss if vloss is not None else tr_loss
        improved = monitor < best * (1 - args.min_delta)   # relative improvement guard
        if monitor < best:
            best = monitor
            torch.save({"model_state_dict": model.state_dict(), "config": vars(args),
                        "epoch": ep, "best_val_loss": best}, ckpt_dir / "best.pt")
        epochs_since_improve = 0 if improved else epochs_since_improve + 1
        # latest.pt every epoch (overwritten) — the canonical "resume/eval me" ckpt
        torch.save({"model_state_dict": model.state_dict(), "config": vars(args),
                    "epoch": ep, "best_val_loss": best}, ckpt_dir / "latest.pt")
        if wb is not None:
            log = {"epoch": ep, "train_loss": tr_loss, "lr": sched.get_last_lr()[0]}
            if vloss is not None:
                log["val_loss"] = vloss; log["best_val_loss"] = best
            wb.log(log)
        es_msg = f" es={epochs_since_improve}/{args.patience}" if args.patience > 0 else ""
        mon = "val" if vloss is not None else "train"
        # log the LR every epoch: the cosine schedule drives the loss-curve shape (a high-LR
        # oscillation plateau early, then a real descent as the LR anneals), and without the LR in
        # the log that shape is easy to misread as convergence. wandb is usually off, so print it.
        print(f"  ep{ep:03d} train={tr_loss:.6f}{vmsg} lr={sched.get_last_lr()[0]:.2e}{es_msg}")
        if (args.patience > 0 and ep + 1 >= args.min_epochs
                and epochs_since_improve >= args.patience):
            print(f"[train] EARLY STOP at ep{ep:03d}: {args.patience} epochs without "
                  f">{args.min_delta*100:.2f}% relative {mon}-loss improvement")
            break
    if wb is not None:
        wb.finish()
    print(f"[train] done -> {ckpt_dir}")


if __name__ == "__main__":
    main()
