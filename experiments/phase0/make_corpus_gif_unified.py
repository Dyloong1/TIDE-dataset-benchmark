"""Unified corpus-GIF generator (publication standard, 2026-06-23).

Produces ONE GIF per corpus config with a STANDARDIZED spec so all 10 are directly
comparable side by side:

  SPEC (fixed for every GIF):
    * panels      : |omega| mid-plane (inferno) + u_x mid-plane (RdBu_r)
    * cadence     : 0.2 T_L per frame  (unified across configs)
    * window      : 10 T_L (50 frames). Decay configs whose available record is
                    shorter use what exists; the title states the actual window.
    * fps/dpi     : 20 fps, 80 dpi
    * colorbar    : per-config 99.5-pctl |omega|, symmetric u_x -- fixed over the
                    clip (no flicker). Per-config scaling is correct (different
                    configs have different Re/eps so absolute |omega| differs).
    * title       : "<id> <case>  t = .. (.. / .. T_L)"  -- shows the real window.

  Reads results/trajectory_showcase_<case>/slices.npz (t, omega, ux, T_L), subsamples
  to the 0.2 T_L cadence, writes <out_dir>/<id>_<case>.gif.

  --delete-slices : remove the source slices.npz after a successful GIF (disk hygiene;
                    the 181 MB slices are render-intermediate, regenerable from ckpts).

Usage:
  python make_corpus_gif_unified.py --case <name> --id <#k> --out <dir> [--delete-slices]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import solver._env  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- unified spec constants ----
CADENCE_TL = 0.2     # T_L per frame
WINDOW_TL = 10.0     # target window (T_L); decay may be shorter -> use available
FPS = 20
DPI = 80


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="case name (trajectory_showcase_<case>)")
    ap.add_argument("--id", required=True, help="catalog id label, e.g. '#7'")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "corpus_gifs"))
    ap.add_argument("--label", default="", help="short physics label for the title")
    ap.add_argument("--delete-slices", action="store_true",
                    help="delete source slices.npz after a successful GIF (disk hygiene)")
    a = ap.parse_args()

    HERE = Path(__file__).parent
    src = HERE / "results" / f"trajectory_showcase_{a.case}" / "slices.npz"
    if not src.exists():
        sys.exit(f"NO slices.npz for {a.case} at {src} (run showcase first)")
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.id.lstrip('#')}_{a.case}.gif"

    z = np.load(src)
    t, om, ux, T_L = z["t"], z["omega"], z["ux"], float(z["T_L"])
    t0 = t[0]
    dt_avail = (t[1] - t[0]) if len(t) > 1 else CADENCE_TL * T_L
    # subsample to the unified 0.2 T_L cadence
    stride = max(1, int(round(CADENCE_TL * T_L / dt_avail)))
    # cap to the 10 T_L window (decay configs may have fewer)
    max_frames = int(round(WINDOW_TL / CADENCE_TL)) + 1   # 51
    idx = list(range(0, len(t), stride))[:max_frames]
    t, om, ux = t[idx], om[idx], ux[idx]
    n = len(t)
    span_TL = (t[-1] - t0) / T_L
    om_max = float(np.percentile(om, 99.5))
    ux_max = float(np.abs(ux).max())
    print(f"{a.id} {a.case}: {n} frames @ {CADENCE_TL} T_L (stride {stride}), "
          f"window {span_TL:.1f} T_L, |omega| clim[0,{om_max:.1f}]")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.0))
    im0 = ax[0].imshow(om[0], cmap="inferno", origin="lower", vmin=0, vmax=om_max,
                       extent=[0, 2 * np.pi, 0, 2 * np.pi])
    ax[0].set(title=r"$|\omega|$ mid-plane", xlabel="x", ylabel="y")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(ux[0], cmap="RdBu_r", origin="lower", vmin=-ux_max, vmax=ux_max,
                       extent=[0, 2 * np.pi, 0, 2 * np.pi])
    ax[1].set(title=r"$u_x$ mid-plane", xlabel="x")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    for axx in ax:
        axx.grid(False)
    sup = fig.suptitle("", fontsize=11)
    fig.tight_layout()

    lab = f" — {a.label}" if a.label else ""

    def update(i):
        im0.set_data(om[i])
        im1.set_data(ux[i])
        sup.set_text(f"{a.id} {a.case}{lab}   256$^3$ fp64   "
                     f"t = {t[i]-t0:5.1f}  ({(t[i]-t0)/T_L:4.1f} / {span_TL:.0f} $T_L$, "
                     f"{CADENCE_TL} $T_L$/frame)")
        return im0, im1, sup

    anim = FuncAnimation(fig, update, frames=n, blit=False)
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    mb = out.stat().st_size / 1e6
    print(f"saved {out} ({mb:.1f} MB)")

    if a.delete_slices:
        # The GIF (the product) is already saved. On Windows the just-read slices.npz can
        # still be briefly locked (WinError 32) -> retry, and if it still won't delete,
        # warn but DO NOT fail: aborting the whole run over a leftover temp file is wrong.
        import time as _time
        for _attempt in range(5):
            try:
                src.unlink()
                print(f"deleted source {src} (disk hygiene)")
                break
            except FileNotFoundError:
                break
            except PermissionError:
                if _attempt == 4:
                    print(f"WARN: could not delete {src} (locked); leaving it. GIF is saved.")
                else:
                    _time.sleep(1.0)


if __name__ == "__main__":
    main()
