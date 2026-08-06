"""Generate train/val/test split manifests for the turbgen DNS corpus (KDD release).

CPU-only, deterministic, data-side tooling. Reads each frame-config's per-config
manifest at  $TURBGEN_DATA_DIR/corpus/<CASE>/manifest.json  (written by
write_corpus_manifest.py) and emits TWO split products into $TURBGEN_DATA_DIR/corpus/:

  1. split_by_ic.json
       WITHIN each config, partition that config's ACCEPTED IC-seeds into
       train/val/test. No seed ever appears in two splits -> this is the
       anti-data-leakage (by-IC) split. A "sample" is a (case, seed) pair;
       all frames of a seed move together (splitting by seed is the point).
       8 seeds -> 6 train / 1 val / 1 test (ratio is a constant below).

  2. split_by_config_ood.json
       Hold out WHOLE configs as an out-of-distribution test set. All seeds of
       a held-out config go to test; every other config is train. The holdout
       set is a constant below with a documented rationale.

The script is robust to missing configs (production is incomplete) and is fully
re-runnable as more configs land: it simply re-reads whatever manifests exist.

    python make_splits.py            # writes both, prints a summary table

Determinism: pure function of the manifests on disk. No randomness, no clock.
Cross-platform: TURBGEN_DATA_DIR env (same default as the other tools); LF only.
"""
import json
import os
import sys
from pathlib import Path

# UTF-8 console (rationale fields / notes may contain non-ascii); harmless on Linux.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SCHEMA_VERSION = "1.0"

# Same cross-platform env pattern as write_corpus_manifest.py / corpus_to_zarr.py.
# Resolves to the Linux T7 path via env at runtime; "D:/turbgen_data" is the
# Windows default for parity.
_ROOT = Path(os.environ.get("TURBGEN_DATA_DIR", "./data"))
CORPUS = _ROOT / "corpus"

# The 15 frame-configs (KDD production spec). Order here is the canonical reporting order.
# NOTE: ou_robust_tau4 DROPPED 2026-06-26 (user decision): marginal config — over 8
# production seeds only 3/8 yielded uniform-100 (38%), 3/8 rejected on A7 stationarity
# (ε fluctuation transiently breaks both k_maxη>=1.5 per-frame AND stationarity). Cannot
# produce a clean uniform-100 16-seed set; τ axis is covered by τ1 + τ2(=ou_relam90).
# NOTE: rotating_ro0p2_v2 ADDED 2026-07-02 (user decision): rotation axis is now 2-point
# — strong (rotating_ro0p2, Ω=2.5, frac_2D=0.97 quasi-2D) + moderate (rotating_ro0p2_v2,
# Ω=0.81, frac_2D~0.68 mixed 2D/3D). So 14 -> 15 frame-configs.
CONFIGS = [
    # forced isotropic HIT (7)
    "ou_relam90_256_fp64",
    "ou_relam70_256_fp64",
    "ou_relam50_256_fp64",
    "helical_re86_retune2_256_fp64",
    "ou_robust_tau1_256_fp64",
    "ou_robust_kf4_256_fp64",
    "ou_robust_kf3_256_fp64",
    # free decay (4)
    "decay_hotstart_re86",
    "decay_saffman_v2",
    "decay_batchelor_v2",
    "abc_turb_full_256_fp64",
    # extended physics (4)
    "rotating_ro0p2_256_fp64",
    "rotating_ro0p2_v2_256_fp64",
    "scalar_sc1_256_fp64",
    "stratified_reb40_256_fp64",
]

# ---- by-IC split policy ------------------------------------------------------
# With the spec's 8 IC-seeds/config: 6 train / 1 val / 1 test. After numeric
# sort of a config's accepted seeds, the FIRST N_TRAIN go to train, the next
# N_VAL to val, the rest to test. Deterministic; degrades gracefully when a
# config currently has fewer accepted seeds (production incomplete).
N_TRAIN = 6
N_VAL = 1
N_TEST = 1
BY_IC_RATIO = f"{N_TRAIN} train / {N_VAL} val / {N_TEST} test (per config, by IC-seed)"

# ---- by-config OOD holdout --------------------------------------------------
# Whole configs held out as an out-of-distribution test set. Easy to change.
# DIRECTION (2026-06-26 user decision): OOD must probe EXTRAPOLATION-TO-HARDER, not
# interpolation-to-easier. So the Re-axis holdout is the HIGHEST Re (relam90), giving
# "train low {55,70} -> test high {86}": the model must extrapolate to MORE small-scale
# structure / wider inertial range / richer high-freq it never saw — the real test for a
# turbulence foundation model (and where NO's known weakness, learning high-freq, bites).
# Holding out the LOWEST Re (old relam50) would be train-hard->test-easy = trivial (just
# express fewer small scales), no discriminative power. relam50 stays in TRAIN.
OOD_HOLDOUT = [
    "stratified_reb40_256_fp64",  # a distinct physics (Boussinesq strat.) absent from train
    "ou_relam90_256_fp64",        # the HIGHEST-Re endpoint -> train low / test high (extrapolate to harder)
]
OOD_RATIONALE = (
    "OOD test = held-out WHOLE configs to probe generalization beyond the training "
    "distribution. 'stratified_reb40_256_fp64' is a different physics regime "
    "(stratified Boussinesq, b-channel) that no training config exhibits, so it tests "
    "transfer to unseen physics. 'ou_relam90_256_fp64' is the HIGHEST-Re endpoint of the "
    "pure-nu Re axis {55,70,86}; holding it out gives train-low/test-high (train {55,70} "
    "-> test 86), testing EXTRAPOLATION to higher Re = more small-scale structure / richer "
    "high-frequency content the model never saw — the discriminative direction (test-harder, "
    "not test-easier). Strictly this is short-range extrapolation (Re_lambda 55->86 < 2x), "
    "reported honestly as such. All seeds of a held-out config go to test; every remaining "
    "config (incl. relam50, the low-Re end, now in TRAIN) is train."
)

LEAKAGE_NOTE = (
    "Frames within a single (case, seed) trajectory are NEVER split across "
    "train/val/test. Consecutive frames of one trajectory are temporally correlated "
    "(0.05 T_L apart), so splitting frames of the same seed across train and test "
    "would leak near-identical states. Splits are always at the (case, seed) "
    "granularity."
)
LEAKAGE_GUARANTEE = (
    "by-IC guarantee: for every config, the train/val/test IC-seed sets are pairwise "
    "disjoint (no seed in two splits). Combined with the per-seed-atomic rule above, "
    "no frame and no IC-seed is shared across splits -> no train/test leakage."
)

# ---- G4: forced -> decay (the diagnosability probe) -------------------------
# The paper's TOP-LEVEL hook (research/07_framing.md) is DIAGNOSABILITY: under a single
# regime you cannot tell whether a model learned the governing PDE or memorized one flow's
# statistics. G4 is the cleanest diagnostic experiment: train on FORCED statistically-
# steady turbulence (an attractor sustained by energy injection) and test on FREE-DECAY
# turbulence (pure Navier-Stokes evolution, no forcing). A model that transfers has learned
# NS structure; one that fails memorized the forced attractor's surface statistics. Both
# regimes are already produced (decay_* configs) — G4 only needs a split (no new data).
# The decay configs are held out WHOLE (all seeds -> test), like the OOD holdout, but on an
# orthogonal axis (forcing presence, not Re/physics-type), so it is a separate product.
DECAY_CONFIGS = [
    "decay_hotstart_re86",
    "decay_saffman_v2",
    "decay_batchelor_v2",
    "abc_turb_full_256_fp64",
]
G4_RATIONALE = (
    "G4 (forced->decay) is the diagnosability probe. TRAIN = all forced/extended configs "
    "(statistically-steady turbulence, an attractor sustained by OU forcing). TEST = the "
    "free-decay configs (decay_hotstart/saffman/batchelor + ABC decay), which evolve under "
    "pure Navier-Stokes with no forcing. A model that transfers forced->decay has learned "
    "NS dynamics; one that only works in-regime memorized the forced attractor's statistics. "
    "Held out on the FORCING-PRESENCE axis (orthogonal to the Re/physics-type OOD holdout). "
    "All seeds of a decay config go to test; every forced/ext config is train. Data already "
    "exists (decay produced), so this split adds no simulation cost. Note: decay is non-"
    "stationary (energy falls by design), so decay eval uses ensemble-per-decay-time metrics, "
    "not cross-time averaging — the benchmark harness applies the decay metric protocol."
)

# ---- G3: cross-forcing-band (k_f {2,3} -> 4) --------------------------------
# G3 holds out the HIGHEST forcing wavenumber on the k_f axis {2,3,4}. TRAIN sees energy
# injected at large scales (k_f=2 anchor ou_relam90 + k_f=3); TEST is k_f=4, where energy is
# injected at a smaller scale -> shifts the injection band closer to the inertial range, a
# shorter large-scale separation the model never saw. NOTE (measured, do NOT call this
# "test-harder like the Re OOD"): raising k_f LOWERS Re_lambda on this grid — measured
# Re_lambda relam90(k_f=2)~86-90 > kf3(k_f=3)=73.4 > kf4(k_f=4)=60.2, and k_maxeta rises
# (1.5 -> 1.637 -> 1.678, i.e. kf4 is the BEST-resolved / lowest-Re end). So the held-out kf4
# is NOT a higher-Re / richer-high-frequency target; the generalization being tested is
# INJECTION-SCALE EXTRAPOLATION (a forcing-band axis orthogonal to Re), not "extrapolate to a
# harder, higher-Re regime". Framing it as the same direction as the Re OOD holdout is wrong.
# Pure code, zero simulation cost (kf4 already produced). Orthogonal to G4 (forcing-presence)
# and the Re/physics OOD (Re axis) — this is the k_f axis. Only OU-forced isotropic configs
# with a defined k_f participate; decay (no forcing) and ext-physics (rotation/scalar/strat,
# whose k_f is the #1 band) are neither train nor test here and are excluded from this split.
G3_KF4_TEST = ["ou_robust_kf4_256_fp64"]
G3_KF_TRAIN = [
    "ou_relam90_256_fp64",       # k_f=2 anchor (the {2,3,4} low end)
    "ou_robust_kf3_256_fp64",    # k_f=3
]
G3_RATIONALE = (
    "G3 (cross-forcing-band) holds out the highest forcing wavenumber on the k_f axis "
    "{2,3,4}. TRAIN = k_f=2 (ou_relam90, the anchor) + k_f=3 (ou_robust_kf3); TEST = k_f=4 "
    "(ou_robust_kf4). Injecting at higher k_f moves the forcing band toward smaller scales / "
    "shortens the large-scale separation, so testing on k_f=4 asks the model to extrapolate the "
    "INJECTION SCALE it never trained on. This is a forcing-wavenumber axis ORTHOGONAL to Re — "
    "and, measured on this grid, raising k_f LOWERS Re_lambda (relam90 ~86-90 > kf3 73.4 > kf4 "
    "60.2, k_maxeta 1.5<1.64<1.68), so held-out kf4 is the lowest-Re / best-resolved end, NOT a "
    "harder higher-Re target. The generalization tested is injection-scale extrapolation, not "
    "'extrapolate to a harder regime' — it is NOT the same direction as the Re OOD holdout (which "
    "does hold out the highest Re). Only OU-forced isotropic configs with a defined k_f take part; "
    "decay (unforced) and ext-physics (rotation/scalar/strat, whose forcing is the #1 band) are "
    "outside this axis and excluded. Data already exists (kf4 produced), so no simulation cost. "
    "All seeds of a config go entirely to its side (per-seed atomic, no leakage)."
)

# ---- G5: cross-physics-type zero-shot (isotropic -> rotation/scalar/stratification) -------
# G5 is the most differentiated OOD axis (no other PDE dataset has a controlled "add-one-
# physics-term" series). TRAIN = the isotropic forced HIT configs (no extra physics term);
# TEST = each extended-physics config zero-shot (rotation / passive scalar / stratification).
# The model must transfer to a governing-equation term it never saw (Coriolis / scalar
# advection-diffusion / buoyancy). CHANNEL ALIGNMENT (honest engineering caveat): scalar/
# stratified add a 5th channel (theta/b); a model trained on 4ch either ignores the extra
# channel (u,v,w,p sub-set transferred) or the harness zero-pads/masks it. This is a
# cross-channel transfer setting, reported as such — not a clean single-variable axis.
# NOTE: scalar/stratified TEST needs the 5th-channel normalizer (theta/b), which is frozen
# only after those configs finish producing; until then G5's scalar/stratified sides are
# flagged empty by the guard (graceful-degrade, same as G3 pre-corpus-merge).
G5_ISO_TRAIN = [
    "ou_relam90_256_fp64", "ou_relam70_256_fp64", "ou_relam50_256_fp64",
    "ou_robust_kf3_256_fp64", "ou_robust_kf4_256_fp64", "ou_robust_tau1_256_fp64",
    "helical_re86_retune2_256_fp64",
]
G5_PHYS_TEST = [
    "rotating_ro0p2_256_fp64", "rotating_ro0p2_v2_256_fp64",
    "scalar_sc1_256_fp64", "stratified_reb40_256_fp64",
]
G5_RATIONALE = (
    "G5 (cross-physics-type zero-shot) is the dataset's most differentiated OOD axis: no "
    "other PDE benchmark ships a controlled 'add one physics term' series. TRAIN = isotropic "
    "forced HIT (no extra term); TEST = extended-physics configs zero-shot (rotation Coriolis "
    "/ passive scalar / stratified Boussinesq). The model must transfer to a governing-equation "
    "term it never trained on. HONEST caveat: scalar/stratified carry a 5th channel (theta/b); "
    "a 4-channel-trained model transfers only the u,v,w,p sub-set (extra channel zero-padded/"
    "masked by the harness) — a cross-channel transfer setting, NOT a clean single-variable "
    "axis, reported as such. Decay is off this axis and excluded. All seeds atomic (no leakage)."
)


def _load_manifests():
    """Return (found, missing): found maps CASE -> manifest dict; missing is a list."""
    found = {}
    missing = []
    for case in CONFIGS:
        mp = CORPUS / case / "manifest.json"
        if mp.exists():
            try:
                found[case] = json.loads(mp.read_text(encoding="utf-8"))
            except Exception as e:  # corrupt manifest: treat as missing, note it
                print(f"[skip] {case}: manifest unreadable ({e})")
                missing.append(case)
        else:
            print(f"[skip] {case}: no manifest at {mp} (not produced yet)")
            missing.append(case)
    return found, missing


def _accepted_seeds(manifest):
    """Sorted numeric list of accepted seeds for a config."""
    seeds = manifest.get("accepted_seeds") or []
    return sorted(int(s) for s in seeds)


def _empty_split_guard(split_name, train_cfgs, test_cfgs, wanted_test, wanted_train=None):
    """Warn + return a flag when a whole-config OOD/G split has an empty/degraded train or test
    side (e.g. a held-out config's manifest isn't on this machine yet, pre-corpus-merge). Mirrors
    the by-IC degraded guard so a silently-empty test set never reads as 'covered everything'.

    wanted_train (optional): the configs the split DECLARES it should train on. If any are
    missing, the train side is silently degraded (e.g. G3's {2,3}->4 collapses to a 1-band
    train when its k_f=2 anchor is absent) — warn on that too, not just the test side.
    Returns (empty: bool, reason: str|None)."""
    missing_test = [c for c in wanted_test if c not in test_cfgs]
    missing_train = [c for c in (wanted_train or []) if c not in train_cfgs]
    empty = (len(test_cfgs) == 0) or (len(train_cfgs) == 0)
    if empty or missing_test or missing_train:
        reason = []
        if len(test_cfgs) == 0:
            reason.append(f"EMPTY test (wanted {wanted_test}; none present — held-out manifest(s) "
                          f"likely on the other machine, populates after corpus merge)")
        elif missing_test:
            reason.append(f"partial test (missing {missing_test} — populates after corpus merge)")
        if len(train_cfgs) == 0:
            reason.append("EMPTY train")
        elif missing_train:
            reason.append(f"DEGRADED train (missing {missing_train} — training on a proper subset "
                          f"of the declared axis; populates after corpus merge)")
        msg = "; ".join(reason)
        print(f"  WARNING [{split_name}]: {msg}")
        return (len(test_cfgs) == 0 or len(train_cfgs) == 0), msg
    return False, None


def _frames_by_seed(manifest):
    """Map seed (int) -> frames_in_corpus (int) from per_seed; default 0."""
    out = {}
    for rec in manifest.get("per_seed", []) or []:
        try:
            out[int(rec["seed"])] = int(rec.get("frames_in_corpus") or 0)
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _partition_by_ic(seeds):
    """Deterministic train/val/test seed lists from sorted seeds.

    Spreads the available seeds across the three splits respecting the
    N_TRAIN/N_VAL/N_TEST policy, but degrades gracefully when fewer seeds exist:
    train is filled first, then val, then test. (With the spec's 8 seeds this is
    exactly 6/1/1.)
    """
    train = seeds[:N_TRAIN]
    val = seeds[N_TRAIN:N_TRAIN + N_VAL]
    test = seeds[N_TRAIN + N_VAL:N_TRAIN + N_VAL + N_TEST]
    # Any seeds beyond N_TRAIN+N_VAL+N_TEST (e.g. a config still carrying >8
    # accepted seeds) fall into test so they are not silently dropped.
    test = test + seeds[N_TRAIN + N_VAL + N_TEST:]
    return train, val, test


def build_by_ic(found):
    per_config = {}
    totals = {"train": 0, "val": 0, "test": 0}
    for case in CONFIGS:
        if case not in found:
            continue
        m = found[case]
        seeds = _accepted_seeds(m)
        fbs = _frames_by_seed(m)
        tr, va, te = _partition_by_ic(seeds)

        def block(seed_list):
            pairs = [[case, s] for s in seed_list]
            frames = sum(fbs.get(s, 0) for s in seed_list)
            return {"seeds": seed_list, "pairs": pairs, "frames": frames}

        b_tr, b_va, b_te = block(tr), block(va), block(te)
        totals["train"] += b_tr["frames"]
        totals["val"] += b_va["frames"]
        totals["test"] += b_te["frames"]
        # A config with <8 accepted seeds yields empty val/test under 6/1/1 -> a
        # by-IC split with no held-out data is a silent leakage/validity hazard
        # (a loader filtering to that config's test gets nothing). Flag it loudly.
        degraded = (len(va) == 0) or (len(te) == 0)
        if degraded:
            print(f"  WARNING: config '{case}' has only {len(seeds)} accepted seeds "
                  f"-> train/val/test = {len(tr)}/{len(va)}/{len(te)} (empty val or test). "
                  f"by-IC split for this config is NOT decision-grade until it has >=8 seeds.")
        # overflow imbalance: seeds beyond 6/1/1 ALL fall into test (see _partition_by_ic),
        # so a config with many seeds trains on only 6 while test balloons (e.g. 21 seeds ->
        # 6/1/14). Not a leakage bug (still per-seed-atomic, disjoint), but the by-IC ratio is
        # skewed: train uses a fixed 6 regardless of how many seeds exist. Flag it so the ratio
        # is honest; raise N_TRAIN if you want more seeds in train.
        overflow = max(0, len(seeds) - (N_TRAIN + N_VAL + N_TEST))
        if overflow:
            print(f"  NOTE: config '{case}' has {len(seeds)} seeds -> {len(tr)}/{len(va)}/{len(te)}"
                  f" (train fixed at {N_TRAIN}; {overflow} overflow seeds all go to test). by-IC"
                  f" ratio is skewed toward test for this config — raise N_TRAIN to rebalance.")
        per_config[case] = {
            "n_accepted_seeds": len(seeds),
            "accepted_seeds": seeds,
            "degraded": degraded,
            "overflow_to_test": overflow,
            "degraded_reason": (f"only {len(seeds)} seeds; empty "
                                + ("val " if len(va) == 0 else "")
                                + ("test" if len(te) == 0 else "")).strip() if degraded else None,
            "train": b_tr,
            "val": b_va,
            "test": b_te,
        }

    n_degraded = sum(1 for c in per_config.values() if c["degraded"])
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "by_ic",
        "policy": "within each config, partition accepted IC-seeds; per-seed-atomic",
        "ratio": BY_IC_RATIO,
        "n_configs_found": len(per_config),
        "n_configs_degraded": n_degraded,
        "total_frames": totals,
        "note": LEAKAGE_NOTE,
        "leakage_guarantee": LEAKAGE_GUARANTEE,
        "degraded_note": ("configs flagged degraded=true have <8 accepted seeds -> "
                          "empty val/test; their by-IC split is NOT decision-grade."),
        "per_config": per_config,
    }


def build_by_config_ood(found):
    train_cfgs, test_cfgs = [], []
    totals = {"train": 0, "test": 0}
    per_config = {}
    for case in CONFIGS:
        if case not in found:
            continue
        m = found[case]
        seeds = _accepted_seeds(m)
        fbs = _frames_by_seed(m)
        frames = sum(fbs.get(s, 0) for s in seeds)
        pairs = [[case, s] for s in seeds]
        is_ood = case in OOD_HOLDOUT
        per_config[case] = {
            "split": "test" if is_ood else "train",
            "n_accepted_seeds": len(seeds),
            "accepted_seeds": seeds,
            "pairs": pairs,
            "frames": frames,
        }
        if is_ood:
            test_cfgs.append(case)
            totals["test"] += frames
        else:
            train_cfgs.append(case)
            totals["train"] += frames

    empty, empty_reason = _empty_split_guard("by_config_ood", train_cfgs, test_cfgs, list(OOD_HOLDOUT))
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "by_config_ood",
        "policy": "hold out whole configs as OOD test; all seeds of a held-out config -> test",
        "ratio": f"train configs / test (OOD) configs: {len(train_cfgs)} / {len(test_cfgs)}",
        "rationale": OOD_RATIONALE,
        "ood_holdout_configs": list(OOD_HOLDOUT),
        "train_configs": train_cfgs,
        "test_configs": test_cfgs,
        "n_configs_found": len(per_config),
        "total_frames": totals,
        "empty_split": empty,
        "empty_split_reason": empty_reason,
        "note": LEAKAGE_NOTE,
        "per_config": per_config,
    }


def build_g4_forced_to_decay(found):
    """G4 diagnosability split: forced/ext configs -> train, decay configs -> test."""
    train_cfgs, test_cfgs = [], []
    totals = {"train": 0, "test": 0}
    per_config = {}
    for case in CONFIGS:
        if case not in found:
            continue
        m = found[case]
        seeds = _accepted_seeds(m)
        fbs = _frames_by_seed(m)
        frames = sum(fbs.get(s, 0) for s in seeds)
        pairs = [[case, s] for s in seeds]
        is_decay = case in DECAY_CONFIGS
        per_config[case] = {
            "split": "test" if is_decay else "train",
            "regime": "free_decay" if is_decay else "forced_or_ext",
            "n_accepted_seeds": len(seeds),
            "accepted_seeds": seeds,
            "pairs": pairs,
            "frames": frames,
        }
        if is_decay:
            test_cfgs.append(case)
            totals["test"] += frames
        else:
            train_cfgs.append(case)
            totals["train"] += frames

    empty, empty_reason = _empty_split_guard("g4_forced_to_decay", train_cfgs, test_cfgs, list(DECAY_CONFIGS))
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "g4_forced_to_decay",
        "policy": "train on forced/ext (steady) configs; test on free-decay configs; "
                  "held out on the forcing-presence axis (diagnosability probe)",
        "ratio": f"train configs / test (decay) configs: {len(train_cfgs)} / {len(test_cfgs)}",
        "rationale": G4_RATIONALE,
        "decay_test_configs": list(DECAY_CONFIGS),
        "train_configs": train_cfgs,
        "test_configs": test_cfgs,
        "n_configs_found": len(per_config),
        "total_frames": totals,
        "empty_split": empty,
        "empty_split_reason": empty_reason,
        "note": LEAKAGE_NOTE,
        "per_config": per_config,
    }


def build_g3_forcing_band(found):
    """G3 cross-forcing-band split: k_f {2,3} -> train, k_f=4 -> test.

    Only the OU-forced isotropic configs with a defined k_f participate; every other
    config (decay, ext-physics) is neither train nor test in this split and is skipped.
    """
    train_cfgs, test_cfgs, skipped = [], [], []
    totals = {"train": 0, "test": 0}
    per_config = {}
    for case in CONFIGS:
        if case not in found:
            continue
        if case in G3_KF_TRAIN:
            role = "train"
        elif case in G3_KF4_TEST:
            role = "test"
        else:
            skipped.append(case)
            continue
        m = found[case]
        seeds = _accepted_seeds(m)
        fbs = _frames_by_seed(m)
        frames = sum(fbs.get(s, 0) for s in seeds)
        per_config[case] = {
            "split": role,
            "k_f": 4 if role == "test" else ("2" if case == "ou_relam90_256_fp64" else "3"),
            "n_accepted_seeds": len(seeds),
            "accepted_seeds": seeds,
            "pairs": [[case, s] for s in seeds],
            "frames": frames,
        }
        (test_cfgs if role == "test" else train_cfgs).append(case)
        totals[role] += frames

    empty, empty_reason = _empty_split_guard("g3_forcing_band", train_cfgs, test_cfgs,
                                             list(G3_KF4_TEST), wanted_train=list(G3_KF_TRAIN))
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "g3_forcing_band",
        "policy": "train on k_f {2,3} forced-isotropic configs; test on k_f=4; held out on the "
                  "forcing-wavenumber axis (injection-scale extrapolation, orthogonal to Re; kf4 "
                  "is the lowest-Re/best-resolved end, NOT test-harder like the Re OOD holdout)",
        "ratio": f"train configs / test configs: {len(train_cfgs)} / {len(test_cfgs)}",
        "rationale": G3_RATIONALE,
        "train_configs": train_cfgs,
        "test_configs": test_cfgs,
        "excluded_configs": skipped,
        "n_configs_found": len(per_config),
        "total_frames": totals,
        "empty_split": empty,
        "empty_split_reason": empty_reason,
        "note": LEAKAGE_NOTE,
        "per_config": per_config,
    }


def build_g5_cross_physics(found):
    """G5 cross-physics-type zero-shot split: isotropic forced HIT -> train, extended-physics
    (rotation/scalar/stratification) -> test. Decay is off-axis and excluded. See G5_RATIONALE
    for the channel-alignment caveat (scalar/stratified carry a 5th channel)."""
    train_cfgs, test_cfgs, skipped = [], [], []
    totals = {"train": 0, "test": 0}
    per_config = {}
    for case in CONFIGS:
        if case not in found:
            continue
        if case in G5_ISO_TRAIN:
            role = "train"
        elif case in G5_PHYS_TEST:
            role = "test"
        else:
            skipped.append(case)
            continue
        m = found[case]
        seeds = _accepted_seeds(m)
        fbs = _frames_by_seed(m)
        frames = sum(fbs.get(s, 0) for s in seeds)
        # channel count for the physics-transfer caveat (scalar/strat = 5ch)
        n_ch = 5 if case in ("scalar_sc1_256_fp64", "stratified_reb40_256_fp64") else 4
        per_config[case] = {
            "split": role,
            "n_channels": n_ch,
            "n_accepted_seeds": len(seeds),
            "accepted_seeds": seeds,
            "pairs": [[case, s] for s in seeds],
            "frames": frames,
        }
        (test_cfgs if role == "test" else train_cfgs).append(case)
        totals[role] += frames

    empty, empty_reason = _empty_split_guard("g5_cross_physics", train_cfgs, test_cfgs,
                                             list(G5_PHYS_TEST), wanted_train=list(G5_ISO_TRAIN))
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "g5_cross_physics",
        "policy": "train on isotropic forced HIT; test on extended-physics configs zero-shot "
                  "(rotation/scalar/stratification); held out on the physics-TYPE axis. "
                  "Cross-channel transfer for the 5ch scalar/stratified tests (see rationale).",
        "ratio": f"train configs / test configs: {len(train_cfgs)} / {len(test_cfgs)}",
        "rationale": G5_RATIONALE,
        "phys_test_configs": list(G5_PHYS_TEST),
        "train_configs": train_cfgs,
        "test_configs": test_cfgs,
        "excluded_configs": skipped,
        "n_configs_found": len(per_config),
        "total_frames": totals,
        "empty_split": empty,
        "empty_split_reason": empty_reason,
        "note": LEAKAGE_NOTE,
        "per_config": per_config,
    }


def _write_json(path, obj):
    # ensure no CR: json.dumps produces \n only; write in binary to be certain.
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))


def _print_summary(by_ic, by_ood):
    n = by_ic["n_configs_found"]
    print()
    print(f"configs found: {n} / {len(CONFIGS)}")
    if n == 0:
        print("0 configs found -- wrote empty split manifests (re-run as configs land).")
        return
    print()
    print("by-IC split (per config: #train/val/test seeds | frames train/val/test):")
    hdr = f"  {'config':<34} {'seeds(t/v/te)':>14}   {'frames(t/v/te)':>20}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for case, c in by_ic["per_config"].items():
        st = len(c["train"]["seeds"])
        sv = len(c["val"]["seeds"])
        se = len(c["test"]["seeds"])
        ft = c["train"]["frames"]
        fv = c["val"]["frames"]
        fe = c["test"]["frames"]
        print(f"  {case:<34} {f'{st}/{sv}/{se}':>14}   {f'{ft}/{fv}/{fe}':>20}")
    t = by_ic["total_frames"]
    tot_frames = "{}/{}/{}".format(t["train"], t["val"], t["test"])
    print(f"  {'TOTAL':<34} {'':>14}   {tot_frames:>20}")
    print()
    print("by-config OOD split:")
    print(f"  train configs ({len(by_ood['train_configs'])}): {', '.join(by_ood['train_configs'])}")
    print(f"  test  configs ({len(by_ood['test_configs'])}): {', '.join(by_ood['test_configs'])}")
    ot = by_ood["total_frames"]
    print(f"  frames  train={ot['train']}  test(OOD)={ot['test']}")


def main():
    found, missing = _load_manifests()
    by_ic = build_by_ic(found)
    by_ood = build_by_config_ood(found)
    by_g4 = build_g4_forced_to_decay(found)
    by_g3 = build_g3_forcing_band(found)
    by_g5 = build_g5_cross_physics(found)

    CORPUS.mkdir(parents=True, exist_ok=True)
    p_ic = CORPUS / "split_by_ic.json"
    p_ood = CORPUS / "split_by_config_ood.json"
    p_g4 = CORPUS / "split_g4_forced_to_decay.json"
    p_g3 = CORPUS / "split_g3_forcing_band.json"
    p_g5 = CORPUS / "split_g5_cross_physics.json"
    _write_json(p_ic, by_ic)
    _write_json(p_ood, by_ood)
    _write_json(p_g4, by_g4)
    _write_json(p_g3, by_g3)
    _write_json(p_g5, by_g5)

    _print_summary(by_ic, by_ood)
    print()
    print("G4 forced->decay split (diagnosability probe):")
    print(f"  train configs ({len(by_g4['train_configs'])}): forced/ext")
    print(f"  test  configs ({len(by_g4['test_configs'])}): {', '.join(by_g4['test_configs'])}")
    gt = by_g4["total_frames"]
    print(f"  frames  train={gt['train']}  test(decay)={gt['test']}")
    print()
    print("G3 cross-forcing-band split (k_f {2,3} -> 4):")
    print(f"  train configs ({len(by_g3['train_configs'])}): {', '.join(by_g3['train_configs'])}")
    print(f"  test  configs ({len(by_g3['test_configs'])}): {', '.join(by_g3['test_configs'])}")
    print(f"  excluded ({len(by_g3['excluded_configs'])}): decay + ext-physics (off-axis)")
    g3t = by_g3["total_frames"]
    print(f"  frames  train={g3t['train']}  test(kf4)={g3t['test']}")
    print()
    print("G5 cross-physics-type zero-shot split (isotropic -> rotation/scalar/strat):")
    print(f"  train configs ({len(by_g5['train_configs'])}): isotropic forced HIT")
    print(f"  test  configs ({len(by_g5['test_configs'])}): {', '.join(by_g5['test_configs'])}")
    print(f"  excluded ({len(by_g5['excluded_configs'])}): decay (off-axis)")
    g5t = by_g5["total_frames"]
    print(f"  frames  train={g5t['train']}  test(ext-phys)={g5t['test']}")
    print()
    print(f"[splits] wrote {p_ic}")
    print(f"[splits] wrote {p_ood}")
    print(f"[splits] wrote {p_g4}")
    print(f"[splits] wrote {p_g3}")
    print(f"[splits] wrote {p_g5}")


if __name__ == "__main__":
    main()
