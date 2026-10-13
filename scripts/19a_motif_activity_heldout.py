"""Amendment 10 part M: motif-based TF activity for PERD-RF on held-out ENCODE tissues (RF only; ENCODE data only).

Motif set S(e): archetypes (with >= 1 activity-scored TF) whose density (matches per kb) z-score across the training
enhancers is >= 2, top 20 by z. Archetype activity = mean ULM activity of its scored TFs. Variants:
  tfmotif = frozen inputs + act:arch for S(e);  tfboth = tfadd inputs + act:arch for S(e).
Same 5 tissue-grouped folds as amendment 05 E; paired tissue-type bootstrap vs frozen and vs tfadd (as 15b).
Usage: python scripts/19a_motif_activity_heldout.py tfmotif|tfboth
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation import tf_activity
from perd_perturbation.perd_rf import run_perd_model

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
TABLES = ROOT / "results" / "tables"
ACT = ROOT / "data" / "processed" / "tf_activity"
SPLIT = ROOT / "data" / "processed" / "tissue_split"
MOTIFS = ROOT / "data" / "processed" / "motifs"
Z_MIN, TOP_N, MIN_KB = 2.0, 20, 0.2
ARCH_PREFIX = "act:arch:"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load_module("split", ROOT / "scripts" / "11d_tissue_grouped_split.py")
boot = load_module("boot", ROOT / "scripts" / "15b_posthoc_tfact_heldout.py")


def archetype_activity(activity: pd.DataFrame) -> pd.DataFrame:
    mapping = pd.read_csv(TABLES / "motif_archetype_tfs.csv")
    rows = {}
    for r in mapping.itertuples():
        tfs = [t for t in str(r.tfs).split(";") if t in activity.index]
        if tfs:
            rows[r.archetype] = activity.loc[tfs].mean(axis=0)
    return pd.DataFrame(rows).T


def motif_sets(enhancers: pd.Index, qualifying: set[str]) -> dict[str, list[str]]:
    counts = pd.read_parquet(MOTIFS / "enhancer_archetypes.parquet")
    counts = counts[counts["archetype"].isin(qualifying) & counts["enhancer_id"].isin(enhancers)]
    gh = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")
    length_kb = ((gh["end"] - gh["start"]) / 1000).clip(lower=MIN_KB)
    counts["density"] = counts["n_matches"] / counts["enhancer_id"].map(length_kb).to_numpy()
    n = len(enhancers)
    stats = counts.groupby("archetype")["density"].agg(["sum", lambda d: (d ** 2).sum()])
    stats.columns = ["s1", "s2"]
    mean = stats["s1"] / n
    sd = np.sqrt((stats["s2"] / n - mean ** 2).clip(lower=1e-12))
    counts["z"] = (counts["density"] - counts["archetype"].map(mean)) / counts["archetype"].map(sd)
    top = counts[counts["z"] >= Z_MIN].sort_values("z", ascending=False).groupby("enhancer_id").head(TOP_N)
    return top.groupby("enhancer_id")["archetype"].apply(list).to_dict()


def main() -> None:
    variant = sys.argv[1]
    if variant not in ("tfmotif", "tfboth"):
        raise SystemExit("usage: 19a_motif_activity_heldout.py tfmotif|tfboth")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    start = time.time()
    rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    frozen_training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    enhancers = pd.Index(frozen_training["enhancer_id"])
    activity = pd.read_parquet(ACT / "activity_encode167.parquet")
    arch_act = archetype_activity(activity)
    arch_act.to_parquet(ACT / "archetype_activity_encode167.parquet")
    sets = motif_sets(enhancers, set(arch_act.index))
    if variant == "tfmotif":
        base = frozen_training[["enhancer_id", "usable_genes"]].copy()
        features = rna
    else:
        binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"]
        targets = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")["target_genes"]
        base = tf_activity.feature_lists(frozen_training, binding, targets, set(activity.index), "tfadd")[["enhancer_id", "usable_genes"]]
        features = tf_activity.with_activity(rna, activity)
    base["usable_genes"] = [list(g) + [ARCH_PREFIX + a for a in sets.get(e, [])] for e, g in zip(base["enhancer_id"], base["usable_genes"])]
    features = pd.concat([features, arch_act.rename(index=lambda a: ARCH_PREFIX + a)[rna.columns]], axis=0)
    n_arch = pd.Series([len(sets.get(e, [])) for e in enhancers])
    info = {"variant": variant, "archetypes_with_activity": int(arch_act.shape[0]), "enhancers_with_motif_inputs": int((n_arch > 0).sum()),
            "motif_inputs_per_enhancer": n_arch.describe().round(2).to_dict(), "inputs_per_enhancer_median": float(base["usable_genes"].map(len).median())}
    print(json.dumps(info, indent=1), f"{time.time() - start:.0f}s", flush=True)

    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    cells = list(rna.columns)
    folds = split.fold_assignment(cells)
    ids = base["enhancer_id"].tolist()
    actual = openness.loc[ids, cells].to_numpy()
    predicted = np.full_like(actual, np.nan, dtype=float)
    for fold in range(split.N_FOLDS):
        train_cells = [c for c, f in zip(cells, folds) if f != fold]
        test_cells = [c for c, f in zip(cells, folds) if f == fold]
        table = run_perd_model(base, features[train_cells], openness[train_cells], features[test_cells], model="rf", n_jobs=n_jobs)
        predicted[:, [cells.index(c) for c in test_cells]] = table.loc[ids, test_cells].to_numpy()
        print(f"{variant} rf fold {fold}: {time.time() - start:.0f}s", flush=True)
    np.save(SPLIT / f"rf_{variant}_out_of_fold.npy", predicted.astype(np.float32))
    info["tissue_grouped_rf"] = split.metrics(actual, predicted)

    types = pd.read_csv(TABLES / "a05_tissue_folds.csv").set_index("cell").loc[cells, "cell_type"].to_numpy()
    tissue_types = np.unique(types)
    cells_of = {t: np.flatnonzero(types == t) for t in tissue_types}
    a32 = actual.astype(np.float32)
    pred = {v: np.load(SPLIT / f"rf{'_' + v if v else ''}_out_of_fold.npy") for v in ("", "tfadd", variant)}
    rng = np.random.default_rng(42)
    draws = [rng.choice(tissue_types, len(tissue_types), replace=True) for _ in range(10_000)]
    rng2 = np.random.default_rng(42)
    small = [rng2.choice(tissue_types, len(tissue_types), replace=True) for _ in range(500)]
    for other in ("", "tfadd"):
        key = f"{variant}_minus_{other or 'frozen'}_rf"
        ra, na = boot.per_cell(a32, pred[variant])
        rb, nb = boot.per_cell(a32, pred[other])
        out = {}
        for label, diff in (("per_cell_r", ra - rb), ("per_cell_nspe", na - nb)):
            per_type = {t: diff[cells_of[t]] for t in tissue_types}
            bs = np.array([np.concatenate([per_type[t] for t in pick]).mean() for pick in draws])
            out[label] = {"mean_diff": float(diff.mean()), "ci": [float(x) for x in np.percentile(bs, [1.25, 98.75])]}
        ca, cb = boot.cross_cell(a32, pred[variant]), boot.cross_cell(a32, pred[other])
        out["cross_cell"] = {"median_r_variant": float(np.nanmedian(ca)), "median_r_other": float(np.nanmedian(cb)),
                             "frac_above_0.5_variant": float(np.nanmean(ca > 0.5)), "frac_above_0.5_other": float(np.nanmean(cb > 0.5))}
        d_med, d_frac = [], []
        for pick in small:
            idx = np.concatenate([cells_of[t] for t in pick])
            x, y = boot.cross_cell(a32[:, idx], pred[variant][:, idx]), boot.cross_cell(a32[:, idx], pred[other][:, idx])
            d_med.append(np.nanmedian(x) - np.nanmedian(y))
            d_frac.append(np.nanmean(x > 0.5) - np.nanmean(y > 0.5))
        out["cross_cell"]["median_r_diff_ci"] = [float(x) for x in np.percentile(d_med, [1.25, 98.75])]
        out["cross_cell"]["frac_diff_ci"] = [float(x) for x in np.percentile(d_frac, [1.25, 98.75])]
        info[key] = out
        print(key, json.dumps(out), flush=True)
    (TABLES / f"a10_motif_heldout_{variant}.json").write_text(json.dumps(info, indent=2, default=float) + "\n")
    print(f"total {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
