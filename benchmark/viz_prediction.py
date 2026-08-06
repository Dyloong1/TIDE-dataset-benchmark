"""viz_prediction.py — save a single-step prediction visualization for a trained rollout baseline.

Loads a trained rollout ckpt, takes one test sample, does a SINGLE-STEP prediction (residual), and
writes a PNG comparing (mid-z velocity-magnitude slice):
    input-last | truth-next | model-pred | persistence | model-error | persistence-error
plus an energy-spectrum panel (truth vs model vs persistence). Lets you eyeball whether the model
learned real dynamics or just copies the last frame (persistence). Self-contained (matplotlib Agg).

  python benchmark/viz_prediction.py --model fno3d --case ou_robust_kf4_256_fp64 \
      --ckpt checkpoints/bench_q/fno3d__rollout__ou_robust_kf4_256_fp64/best.pt --out viz/fno3d_kf4.png
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Model.baselines import get_model  # noqa: E402
from Dataset.turbgen_zarr_dataset import TurbgenZarrDataset  # noqa: E402
from train_benchmark import _predict  # noqa: E402
import benchmark_metrics as BM  # noqa: E402


def _vmag(f):  # velocity magnitude of a (C,N,N,N) frame -> (N,N,N)
    return f[:3].pow(2).sum(0).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default=os.path.join(os.environ.get("TURBGEN_DATA_DIR", ""), "corpus", "benchmark_slice.json"))
    ap.add_argument("--data-root", default=os.environ.get("TURBGEN_DATA_DIR", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=28)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--no-residual", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = TurbgenZarrDataset(case=args.case, split="test", slice_manifest=args.slice, task="rollout",
                            t_in=5, t_out=1, data_root=args.data_root, max_samples=args.sample + 1)
    s = ds[args.sample]
    x = s["input_dense"].unsqueeze(0).to(dev)   # (1,5,C,N,N,N)
    y = s["output_dense"].unsqueeze(0).to(dev)  # (1,1,C,N,N,N)
    C = x.shape[2]

    inds = ['u', 'v', 'w', 'p'][:C] if C <= 4 else ['c%d' % i for i in range(C)]
    cfg = SimpleNamespace(INPUT_TIMESTEPS=5, OUTPUT_TIMESTEPS=1, INDICATORS=inds, IN_CHANNELS=C, OUT_CHANNELS=C,
                          FNO_WIDTH=args.width, UFNO_WIDTH=args.width, FNO_LAYERS=args.layers,
                          FNO_MODES_Z=args.modes, FNO_MODES_H=args.modes, FNO_MODES_W=args.modes,
                          TFNO_RANK=16, DEEPONET_P=128, DEEPONET_WIDTH=args.width)
    m = get_model(args.model, cfg).to(dev).eval()
    sd = torch.load(args.ckpt, map_location=dev)
    m.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    with torch.no_grad():
        pred = _predict(m, x, not args.no_residual)   # (1,1,C,N,N,N)

    last = x[:, -1:]                                    # persistence = last input frame
    N = x.shape[-1]; z = N // 2
    truth_f = y[0, 0].cpu(); pred_f = pred[0, 0].cpu(); pers_f = last[0, 0].cpu(); inp_f = x[0, -1].cpu()
    nr_m = float((pred_f - truth_f).norm() / truth_f.norm())
    nr_p = float((pers_f - truth_f).norm() / truth_f.norm())

    panels = [("input (last)", _vmag(inp_f)[z]), ("truth (next)", _vmag(truth_f)[z]),
              (f"model  nRMSE={nr_m:.3f}", _vmag(pred_f)[z]),
              (f"persist nRMSE={nr_p:.3f}", _vmag(pers_f)[z]),
              ("|model-truth|", (_vmag(pred_f)[z] - _vmag(truth_f)[z]).abs()),
              ("|persist-truth|", (_vmag(pers_f)[z] - _vmag(truth_f)[z]).abs())]
    vmax = max(float(_vmag(truth_f)[z].max()), float(_vmag(pred_f)[z].max()))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, (title, img) in enumerate(panels):
        ax = axes[i // 4][i % 4]
        vm = vmax if "|" not in title else None
        im = ax.imshow(img.numpy(), cmap="turbo" if "|" not in title else "inferno", vmin=0, vmax=vm)
        ax.set_title(title, fontsize=11); ax.axis("off"); plt.colorbar(im, ax=ax, fraction=0.046)
    # energy spectrum panel (velocity)
    axk = axes[1][2]
    ks, Et = BM.energy_spectrum(truth_f[:3]); _, Ep = BM.energy_spectrum(pred_f[:3]); _, Es = BM.energy_spectrum(pers_f[:3])
    axk.loglog(ks.cpu(), Et.cpu(), 'k-', label='truth', lw=2)
    axk.loglog(ks.cpu(), Ep.cpu(), 'r--', label='model', lw=1.5)
    axk.loglog(ks.cpu(), Es.cpu(), 'b:', label='persist', lw=1.5)
    axk.set_title("energy spectrum E(k)", fontsize=11); axk.legend(fontsize=9); axk.set_xlabel("k"); axk.grid(alpha=0.3)
    axes[1][3].axis("off")
    axes[1][3].text(0.05, 0.5,
                    f"model: {args.model}\ncase: {args.case}\nsingle-step (Δt=0.05 T_L)\n"
                    f"residual: {not args.no_residual}\n\nmodel nRMSE:   {nr_m:.4f}\n"
                    f"persist nRMSE: {nr_p:.4f}\n\n{'MODEL BEATS persist' if nr_m < nr_p - 0.005 else 'model ≈ persist (no gain)'}",
                    fontsize=12, family="monospace", va="center")
    fig.suptitle(f"{args.model} single-step prediction — {args.case}", fontsize=14)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=90, bbox_inches="tight"); plt.close()
    print(f"[viz] wrote {args.out}  (model nRMSE {nr_m:.4f} vs persist {nr_p:.4f})")


if __name__ == "__main__":
    main()
