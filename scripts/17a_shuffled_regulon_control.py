"""Amendment 09 A: shuffled-regulon control for the TF-activity gain on held-out ENCODE tissues. ENCODE data only.

Null regulon: per TF keep the number of targets and their signs, replace targets by random reference genes (seed 42).
Activity (ULM) for the 167 ENCODE cells on the same 766 TFs; "Add" features; tissue-grouped 5-fold (as 14c);
paired bootstrap tfadd - tfshuf and tfshuf - frozen (as 15b).
"""

from __future__ import annotations

import importlib.util
import json
import os
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load_module("split", ROOT / "scripts" / "11d_tissue_grouped_split.py")
boot = load_module("boot", ROOT / "scripts" / "15b_posthoc_tfact_heldout.py")


def shuffled_network(network: pd.DataFrame, genes: pd.Index, tfs: pd.Index, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gene_array = genes.to_numpy()
    rows = []
    for tf, group in network[network["source"].isin(tfs)].groupby("source", sort=True):
        n = int(group["target"].isin(genes).sum())
        signs = group.loc[group["target"].isin(genes), "weight"].to_numpy()
        targets = rng.choice(gene_array, size=n, replace=False)
        rows.append(pd.DataFrame({"source": tf, "target": targets, "weight": signs}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    start = time.time()
    raw = pd.read_csv(ROOT / "data" / "raw" / "collectri" / "collectri.tsv", sep="\t")
    network = tf_activity.network_from_collectri(raw)
    rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    mean, sd = tf_activity.reference_scale(rna)
    real_act = pd.read_parquet(ACT / "activity_encode167.parquet")
    null = shuffled_network(network, mean.index, real_act.index)
    weights = tf_activity.weight_matrix(null, mean.index)
    weights = weights.reindex(columns=real_act.index).fillna(0.0)
    act = tf_activity.activity(rna, mean, sd, weights)
    act.to_parquet(ACT / "activity_encode167_shuffled.parquet")
    nr3c1_corr = float(np.corrcoef(act.loc["NR3C1"], real_act.loc["NR3C1"])[0, 1])
    print(f"shuffled regulon: {len(null):,} edges, {weights.shape[1]} TFs; NR3C1 real-vs-shuffled activity r = {nr3c1_corr:.3f}; {time.time() - start:.0f}s", flush=True)

    frozen_training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"]
    targets = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")["target_genes"]
    training = tf_activity.feature_lists(frozen_training, binding, targets, set(act.index), "tfadd")
    features = tf_activity.with_activity(rna, act)
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    cells = list(rna.columns)
    folds = split.fold_assignment(cells)
    enhancers = training["enhancer_id"].tolist()
    actual = openness.loc[enhancers, cells].to_numpy()
    results = {"nr3c1_real_vs_shuffled_activity_r": nr3c1_corr}
    for model in ("ridge", "rf"):
        predicted = np.full_like(actual, np.nan, dtype=float)
        t0 = time.time()
        for fold in range(split.N_FOLDS):
            train_cells = [c for c, f in zip(cells, folds) if f != fold]
            test_cells = [c for c, f in zip(cells, folds) if f == fold]
            table = run_perd_model(training, features[train_cells], openness[train_cells], features[test_cells], model=model, n_jobs=n_jobs)
            predicted[:, [cells.index(c) for c in test_cells]] = table.loc[enhancers, test_cells].to_numpy()
            print(f"tfshuf {model} fold {fold}: {time.time() - t0:.0f}s", flush=True)
        np.save(SPLIT / f"{model}_tfshuf_out_of_fold.npy", predicted.astype(np.float32))
        results[f"tissue_grouped_{model}"] = split.metrics(actual, predicted)

    # Paired bootstrap over tissue types (as 15b).
    types = pd.read_csv(TABLES / "a05_tissue_folds.csv").set_index("cell").loc[cells, "cell_type"].to_numpy()
    tissue_types = np.unique(types)
    cells_of = {t: np.flatnonzero(types == t) for t in tissue_types}
    actual32 = actual.astype(np.float32)
    pred = {v: {m: np.load(SPLIT / f"{m}{'_' + v if v else ''}_out_of_fold.npy") for m in ("rf", "ridge")} for v in ("", "tfadd", "tfshuf")}
    rng = np.random.default_rng(42)
    draws = [rng.choice(tissue_types, len(tissue_types), replace=True) for _ in range(10_000)]
    rng2 = np.random.default_rng(42)
    small = [rng2.choice(tissue_types, len(tissue_types), replace=True) for _ in range(500)]
    for a, b in (("tfadd", "tfshuf"), ("tfshuf", "")):
        for m in ("rf", "ridge"):
            key = f"{a}_minus_{b or 'frozen'}_{m}"
            ra, na = boot.per_cell(actual32, pred[a][m])
            rb, nb = boot.per_cell(actual32, pred[b][m])
            out = {}
            for label, diff in (("per_cell_r", ra - rb), ("per_cell_nspe", na - nb)):
                per_type = {t: diff[cells_of[t]] for t in tissue_types}
                bs = np.array([np.concatenate([per_type[t] for t in pick]).mean() for pick in draws])
                out[label] = {"mean_diff": float(diff.mean()), "ci": [float(x) for x in np.percentile(bs, [1.25, 98.75])]}
            ca, cb = boot.cross_cell(actual32, pred[a][m]), boot.cross_cell(actual32, pred[b][m])
            out["cross_cell_median_r"] = {"a": float(np.nanmedian(ca)), "b": float(np.nanmedian(cb)), "frac_above_0.5_a": float(np.nanmean(ca > 0.5)), "frac_above_0.5_b": float(np.nanmean(cb > 0.5))}
            d_med, d_frac = [], []
            for pick in small:
                idx = np.concatenate([cells_of[t] for t in pick])
                x, y = boot.cross_cell(actual32[:, idx], pred[a][m][:, idx]), boot.cross_cell(actual32[:, idx], pred[b][m][:, idx])
                d_med.append(np.nanmedian(x) - np.nanmedian(y))
                d_frac.append(np.nanmean(x > 0.5) - np.nanmean(y > 0.5))
            out["cross_cell_median_r"]["diff_ci"] = [float(x) for x in np.percentile(d_med, [1.25, 98.75])]
            out["cross_cell_median_r"]["frac_diff_ci"] = [float(x) for x in np.percentile(d_frac, [1.25, 98.75])]
            results[key] = out
            print(key, json.dumps(out), flush=True)
    (TABLES / "a09_shuffled_regulon_control.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"total {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
