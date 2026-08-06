# Datasheet — 256³ DNS Turbulence Dataset (turbgen)

Datasheet following Gebru et al. (2021). Every quantitative claim here is grounded
in the per-case acceptance appendices (`experiments/phase0/results/<case>/DNS_STANDARD_APPENDIX_A.md`)
and the production catalog; every report carries a provenance stamp
(git hash / date / machine / torch version) and full verification tables.

---

## 1. Motivation

- **Why was this dataset created?** To provide a corpus of 3D incompressible
  turbulence for training and benchmarking turbulence models, where every
  configuration is verified against an objective DNS acceptance standard (resolution
  class, A-group hard gates, and a D-group equation-level consistency check).
- **What it adds relative to common practice.** Beyond the usual statistics
  (spectrum, isotropy), each configuration is checked with (a) a D-group verifying
  the field satisfies the Navier–Stokes equations to time-truncation precision, and
  (b) high-variance statistics (e.g. A10 isotropy) reported as ensemble mean ± std
  over an IC-ensemble (multiple initial conditions; see Uses for the honest scope of
  "independent") rather than a single value.

## 2. Composition

- **Solver**: GPU pseudo-spectral DNS of the incompressible Navier–Stokes equations
  (rotational form), 256³, 2/3-rule dealiasing, Williamson RK3 + Lawson integrating
  factor for viscosity, **fp64 compute**. Forcing for steady cases is Eswaran–Pope
  stochastic Ornstein–Uhlenbeck (random-phase, low-band) — the only forcing that
  avoids the metastable relaminarization of deterministic band forcing on 256³
  (documented negative result, `NEGATIVE_RESULT_deterministic_forcing.md`).

- **Configurations**: **15 released frame-configs** spanning controlled physics axes
  (pure-ν Re, forcing-band k_f, correlation-time τ, rotation, passive scalar, stratification,
  helicity, free decay), plus boundary/validation anchors (NOT released as corpus). All 256³, fp64.
  The authoritative config list is `experiments/phase0/make_splits.py::CONFIGS`.

  **Released frame-configs (15):**

  | # | name | axis / type | Re_λ | class | verdict |
  |---|---|---|---|---|---|
  | #1 | ou_relam90 | Re axis (k_f=2, τ=2 anchor) | 86 | I | A+D all-green (14 seeds) |
  | #2 | ou_relam70 | Re axis | 70 | I | A+D all-green (dt=0.05: 16 seeds/2416 fr, A10 cross 1.43%) |
  | Re50 | ou_relam50 | Re axis (low end, meas. 54.8) | 55 | I | A+D all-green (14 seeds) |
  | kf3 | ou_robust_kf3 | k_f axis (k_f=3) | 73 | I | A+D all-green (15 seeds) |
  | kf4 | ou_robust_kf4 | k_f axis (k_f=4) | 60 | I | A+D all-green (16 seeds) |
  | τ1 | ou_robust_tau1 | τ axis (τ=1) | 80 | I | A+D all-green (14 seeds) |
  | #7 | helical_re86 | helicity (broken mirror symmetry) | 83 | I | A+D all-green (13 seeds) |
  | decay_hotstart | decay_hotstart_re86 | free decay (hot start) | 86→24 | — | decay-criteria + D all-green (**8 independent trajectories, v2 2026-07-15**) |
  | decay_saffman | decay_saffman_v2 | free decay (initial spectrum p=2) | cold start | — | decay-criteria + D all-green (8 seeds) |
  | decay_batchelor | decay_batchelor_v2 | free decay (initial spectrum p=4) | cold start | — | decay-criteria + D all-green (8 seeds) |
  | ABC | abc_turb_full | helical decay (Beltrami init) | 230→13 | — | decay-criteria + D all-green (8 seeds) |
  | rot-strong | rotating_ro0p2 | rotation (Ω=2.5, frac_2D 0.95) | — | I | hard gate (res+D) pass; A10 report-only (8 seeds, cross 0.36%) |
  | rot-moderate | rotating_ro0p2_v2 | rotation (Ω=0.81, frac_2D 0.64) | — | I | hard gate (res+D) pass; A10 report-only (8 seeds, cross 1.41%) |
  | scalar | scalar_sc1 | passive scalar (Sc=1, 5th channel θ) | 86 | I | hard gate (res+Batchelor+D) pass; A10 report-only (8 seeds, cross 3.31%) |
  | stratified | stratified_reb40 | stratification (Boussinesq, Re_b≈41, 5th channel b) | 86 | I | **complete**; hard gate (res+Re_b+Ozmidov+D) pass; A10 report-only (8 seeds, cross 0.42%) |

> **RESOLVED DEFECT (kept for provenance) — `decay_hotstart_re86` originally shipped 8 bit-identical
> copies of one trajectory** (verified 2026-07-15: `max|diff|=0.0` across all frames, shared ckpt md5,
> N_eff≈1.0; cause: 8 pools resumed the SAME checkpoint under different seed labels, and
> `corpus_to_zarr.py` counted directories). **Re-produced same day as v2: 8 genuinely independent
> trajectories** warm-started from 8 DISTINCT `ou_relam90` checkpoints (md5 8/8 unique, first-frame
> pairwise `max|diff| ≥ 2.55`, DEC-criteria 8/8 green, zarr attrs now carry
> `n_independent_trajectories=8`). The defective 79G corpus was deleted with explicit user
> authorization after v2 passed acceptance. The full record is retained in the production ledger.
> **All 15 configs now have machine-verified independent seeds**
> (`benchmark/tests/test_corpus_seed_independence.py`, KNOWN_DUPLICATE exemption list now empty).

> **KNOWN LIMITATION — normalizer constants are transductively fitted.** The per-group minmax
> scalars in each `<CASE>_norm_minmax.json` were fitted over EVERY frame in the corpus, including
> val/test seeds (`corpus_to_zarr.py` never filters by seed). The leak is a single per-group global
> extremum — not per-sample statistics — and was measured to shift the velocity scalar by ~7% on one
> config. Fix requires refitting + retraining everything; judged not worth it. Papers using this
> dataset must describe the constants as "corpus-global frozen scales", NOT "train-only".

  **Boundary / validation anchors (NOT released as corpus — honest boundary evidence):**

  | # | name | role | note |
  |---|---|---|---|
  | #4 | ou_relam100 | Class II resolution boundary | Re_λ111; A2/A4 fail (under-resolved); spectrum+low-order+D usable |
  | #A2 | ou_relam95_classII | Class II resolution boundary | Re_λ98; A2=99.47% marginal; densifies the Class II degradation curve |
  | ~~tau4~~ | ou_robust_tau4 | admitted at config level, **DROPPED from release** | τ=4 passed config-admission (long-trajectory 4-seed pooled: A13/13 + D-group, k_maxη=1.551 Class I; see ledger1) but was DROPPED from the released frame-set (2026-06-26): only 3/8 seeds reached uniform-100 frames, rejects all A7 stationarity. Released τ axis is 2 points {1,2}. |
  | TG | tg_re1600 | Brachet validation anchor | deterministic symmetric decay; not corpus (half of A-group N/A); ε(t) peak matches Brachet |

- **Corpus (15/15 complete, dt=0.05, 2026-07-13)**: frame-configs × independent IC-ensemble
  seeds × **150 frames per seed** for forced/extended (decay configs span the active-decay
  window). **Frame cadence = 0.05 T_L (~1 τ_η)** (2026-07-08 dt redesign: the old 0.2 T_L
  cadence decorrelated adjacent frames and degenerated the rollout task; 0.05 T_L makes the
  adjacent-frame change ~17%, learnable — aligns with JHTDB/The Well/APEBench). **Each seed
  carries its OWN independent OU forcing sequence** (`ou_seed = config + IC seed`), so the
  ensemble is a fully-independent ensemble — see §5. Per-config released seed/frame counts
  (sized so the pooled A10 clears threshold; the k_f=2 high-Re cases need more seeds to beat
  the few-mode uw floor): ou_relam90 12/2427, ou_relam70 16/2416 (A10 cross 1.427%),
  ou_relam50 8/1200, kf4/kf3 8/1200, tau1 10/1500, helical 8/1200, rotating×2 8/1200,
  scalar 8/1089, stratified 8/1200, hotstart 8/400, decay-family per decay span. Each frame
  is a 4-channel (u, v, w, p) fp32 field of shape [4, 256, 256, 256] (scalar/stratified add a
  5th channel θ/b). Pressure is solved by the D4-certified `operators.pressure_hat`. Each
  frame is **instantaneously Class I** (per-frame k_maxη≥1.5 gate). Data currently split
  across the production machines — see
  `DATASET_MANIFEST_dt0.05.md`; merged to a single staging root at release.

- **Labels / derived quantities**: each frame stores its time `t` and instantaneous
  `k_max_eta`. Per-case appendices give the full A/B/D acceptance tables.

## 3. Collection process

- Each configuration: spin-up to statistical stationarity (or, for decay, warm-start
  from a matured field), then a sampling window. Acceptance is computed by the audited
  referee scripts `eval_dns_standard.py` (A/B) + `eval_dynamics_residuals.py` (D);
  the referee scripts themselves pass known-answer target tests (`test_eval_acceptance.py`,
  including adversarial cases that must FAIL) before judging data.
- Corpus frames come from a triple gate: per-frame (instantaneous k_maxη≥1.5),
  per-trajectory (class/A2/A12/A14/D1–D4), and multi-seed pooling (A10/A7 time-averaged
  items). Only frames from accepted trajectories enter the corpus.

## 4. Preprocessing / cleaning / labeling

- **Compute fp64, store fp32**: storing an already-accepted fp64 field as fp32 is
  verified safe (A12 incompressibility ~1e-13 after the round trip). fp32 compute is
  NOT used (it has a structural small-scale instability).
- **Per-frame resolution filtering**: frames whose instantaneous k_maxη<1.5 are
  dropped. **Consequence (declared in manifest)**: the dropped frames are the eps-peak
  (most intermittent) instants, so the corpus systematically under-samples the
  strong-dissipation tail; high-order intermittency statistics (derivative
  flatness/kurtosis) are a **lower bound**, not unbiased. Mild at Re_λ≈86 (dropped eps
  ~1.3–1.4× the mean). Frames are therefore NOT uniformly spaced in time — loaders MUST
  use each frame's stored `t`.

## 5. Uses

- **Recommended**: training/benchmarking turbulence foundation models on resolved 3D
  incompressible turbulence; testing equation-level consistency (D-group); studying
  forced-steady vs free-decay and non-helical vs helical regimes.
- **Use with care / not recommended**:
  - Do NOT treat the corpus as an unbiased sample of extreme dissipation events (see §4).
  - Do NOT cite the free-decay decay-rate α as a quantitative physical result: α is a
    **report-only** item — it depends strongly on the virtual-origin t0 (a free fit
    parameter; in-window t0 shifts move α across ~1.3–2.0 at R²>0.998) and deviates
    from textbook Saffman/Batchelor values. The decay cases' defensible value is the
    D-group (equation-level correctness), not α.
  - Do NOT judge isotropy (A10) from a single realization: A10 cross-term is a
    **high-variance** statistic (per-seed scatter ~0.7–6%); the verdict uses the
    multi-seed **pooled** value (corpus: cross 1.92%, comp 2.27%) and is reported as
    ensemble mean ± std (corpus per-seed cross 2.83% ± 1.39%). This pooling is the DNS
    standard's own A9/reporting rule ("average ≥5 T_E **or** ≥8 independent ensemble
    samples; judge on the mean, report uncertainty"; Pope 2000) — pooling reduces the
    finite-volume statistical noise of an estimate of a quantity that is ~0 by isotropy,
    it does not "wash" an anisotropic field isotropic (a truly anisotropic field's
    pooled value would not drop). All healthy seeds are kept (only laminarized /
    A14-aborted runs are excluded), thresholds are unchanged, and per-seed scatter is
    reported alongside the pooled value.
  - **Independence of ensemble seeds**: each seed varies BOTH the initial condition
    (`--seed`) AND the OU forcing sequence — `ou_seed = config + IC seed` gives every
    seed its own independent stochastic forcing drive (`--ou-seed-per-ic`). So the
    ensemble is a **fully-independent** ensemble, not merely IC-varied. This is the core
    of the A10 fix: independent forcing randomizes the sign of the cross-correlations
    ⟨u_i u_j⟩ across seeds, so signed pooling cancels them to the isotropic value. (An
    earlier release shared one fixed `ou_seed` across seeds; that imprinted a common
    directional bias that signed pooling could not cancel — corrected to per-seed
    forcing.)
  - #4 (Re_λ111) is Class II: gradient/high-order statistics are N/A (under-resolved);
    spectrum, low-order moments, and D-group are usable. It is a measured point on the
    256³ Class I boundary: the publishable-Re ceiling (~Re_λ 86–90) is not a chosen limit
    but the objective consequence of the DNS resolution requirement — at fixed N=256 the
    Class I red line k_maxη≥1.5 caps Re, and #4 (Re111, A2 dissipation-fraction fails) plus
    ou_relam95 (Re98, A2 marginally fails) are where that line measurably sits. Higher Re is
    physically reachable only as Class II (spectrum + low-order + D usable, gradients N/A).

## 6. Distribution

- Format: **zarr only** (lossless zstd) — `.pt` frames are an intermediate, deleted after
  conversion + QA. `corpus/<case>/<case>.zarr` holds u[N,3,256,256,256] + p[N,256,256,256]
  (+ θ or b as a 5th field for scalar/stratified) + t/seed/k_max_eta arrays (1 frame/chunk).
  Storage root is the `TURBGEN_DATA_DIR` env var (not a hardcoded path), so the tooling is
  cross-platform. Frozen JOINT-velocity normalizers in `<case>_norm_minmax.json` (maps to
  [-1,1]) and `<case>_norm_standardize.json` (z-score, UNBOUNDED ~±5, pressure to ~−11);
  velocity is normalized by ONE shared scale (not per-component) to preserve A10 isotropy.
  Lossless verified bit-exact (random frame spot-checks). `seed` index orders seeds numerically.
- Hosting: full corpus on **Zenodo (DOI)**, core subset on **HuggingFace**; data **CC-BY-4.0**,
  code **MIT**.
- License / public release: **pending user decision** (publication and public release
  require maintainer sign-off).

### Reproducibility scope (honest)
- **Bit-exact regeneration requires the ORIGINAL (device, torch version).** The OU forcing's
  random sequence is generated on-GPU; GPU RNG differs from CPU and can differ across GPU
  architectures / torch versions. The two production machines run different torch (2.11 vs
  2.7), so a trajectory is byte-identical only when re-run on the same hardware+torch that
  produced it. **The released fp32 zarr is the canonical bit-exact artifact.**
- From the public solver + config + seed + git hash you can **regenerate the same flow
  STATISTICS** (spectra, A/B/D-group verdicts) on any fp64-capable device — the dataset's
  statistical properties are reproducible everywhere; per-byte identity is hardware-bound.
  This is the standard situation for GPU-RNG DNS and is stated so reviewers reproducing on
  different hardware expect matching statistics, not matching bytes.

## 7. Maintenance

- **Paper:** https://arxiv.org/abs/2608.04222
- **Maintainer / contact:** Yilong Dai (<ydai17@ua.edu>). Errata and additions are
  released as new dataset versions under the concept DOI
  (10.5281/zenodo.21589489); issues and questions go to the GitHub tracker
  (https://github.com/Dyloong1/TIDE-dataset-benchmark/issues).

- Two-machine generation (Windows/torch 2.11 and Linux/torch 2.7) sharing one
  git repo; the solver is cross-machine reproduced (#1 independently reproduced on both
  OS / torch versions, A+D all-green, consistent). Acceptance appendices carry
  provenance stamps for traceability.

## 8. Known limitations / open items

- **Production scope:** 15 frame-configs × **8–16 IC-seeds (forced) or 8 (ext/decay)** × 150
  frames (decay configs span the active-decay window, frozen tail excluded), zarr-only.
  **As of 2026-07-13 production is 15/15 complete** (all forced/ext/decay accepted and on
  disk). Per-config accepted-seed counts vary with the per-trajectory gate and the pooled-A10
  requirement (forced k_f=2 high-Re cases pool more seeds: relam90 12, relam70 16; others 8–10
  — see the released-frame-set table §2 and `DATASET_MANIFEST_dt0.05.md`). Data is split
  across the production machines; merged to a single staging root at release.
- **Regime framing is honest, not orthogonal axes:** only the pure-ν Re axis {Re_λ 55,70,86}
  is single-variable (the low point measured Re_λ=54.8, nominal "50"). The k_f {2,3,4} and τ {1,2} variations are physically-distinct
  forcing regimes but are CONFOUNDED with Re (higher-k injection / longer correlation shift
  ε → Re); the manifest records each config's MEASURED (Re_λ, k_f, τ, ε). Physical diversity
  comes from multi-physics breadth + this measured-regime coverage, not a wide clean sweep.
- **Frame count after the per-frame gate:** the instantaneous k_maxη≥1.5 gate drops some
  frames (Re86-edge configs like scalar drop ~half on the ε-peak instants); production
  over-provisions the sampling window so enough frames survive the gate. Frame counts are
  therefore NOT uniform across seeds (accepted as of 2026-07-04); per-seed kept counts are in
  the manifest.
- **Re_λ≈55 caveat (the Re-axis low point, config `ou_relam50`):** a low-Re /
  dissipation-dominated regime with essentially no inertial range (B-group is report-only
  below Re_λ100); included as the Re-axis low endpoint, not as an inertial-range-physics
  sample. Calibrated 2026-06-25: σ²=0.0555 → Re_λ=54.8, A5 L_box/L=5.35 PASS (the flagged
  A5 risk did not materialize). Named "50" (nominal), measured 54.8.
- #4 D2 re-measured under the frozen protocol (2026-06-21): mean 1.04e-3 (was
  3.96e-3 live-forcing); verdict unchanged. All D2 values now use the frozen protocol.
