"""Amendment 05 E: PERD accuracy on held-out *tissues* (5-fold, grouped by ENCODE cell type). ENCODE data only.

Per fold, each enhancer's PERD-ridge and PERD-RF model (same genes and settings as section 7) is fit on the
training cells and predicts the held-out cells. Metrics on the assembled out-of-fold matrix: cross-cell Pearson
per enhancer, cross-enhancer Pearson per cell, pooled NSPE. The paper-style random split from ../perd
(netopen_rf_predictions, 57 test cells) is recomputed with the same code for comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation.vendor.perd_repro.cell_types import extract_cell_type
from perd_perturbation.perd_rf import run_perd_model


ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "data" / "processed" / "tissue_split"
N_FOLDS = 5


def fold_assignment(cells: list[str]) -> pd.Series:
    """Groups (cell types) shuffled with seed 42, each assigned to the fold with the fewest cells so far."""
    types = pd.Series({c: extract_cell_type(c) for c in cells})
    sizes = types.value_counts()
    order = list(sizes.index)
    np.random.default_rng(42).shuffle(order)
    load, fold_of_type = np.zeros(N_FOLDS), {}
    for t in order:
        f = int(np.argmin(load))
        fold_of_type[t] = f
        load[f] += sizes[t]
    return types.map(fold_of_type)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """actual/predicted: enhancers x cells."""
    cross_cell = []
    for a, p in zip(actual, predicted):
        if np.std(a) > 0 and np.std(p) > 0:
            cross_cell.append(np.corrcoef(a, p)[0, 1])
    cross_enhancer = [np.corrcoef(actual[:, j], predicted[:, j])[0, 1] for j in range(actual.shape[1])]
    pooled = np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2)
    cross_cell = np.asarray(cross_cell)
    return {
        "cross_cell_pearson_median": float(np.median(cross_cell)),
        "cross_cell_pearson_mean": float(np.mean(cross_cell)),
        "fraction_enhancers_cross_cell_above_0.5": float(np.mean(cross_cell > 0.5)),
        "cross_enhancer_pearson_median": float(np.median(cross_enhancer)),
        "pooled_nspe": float(pooled),
        "n_enhancers": int(actual.shape[0]),
        "n_cells": int(actual.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Debug: only the first N enhancers")
    parser.add_argument("--models", nargs="+", default=["ridge", "rf"])
    args = parser.parse_args()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    OUT.mkdir(parents=True, exist_ok=True)
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    if args.limit:
        training = training.head(args.limit)
    rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    cells = list(rna.columns)
    folds = fold_assignment(cells)
    fold_table = pd.DataFrame({"cell": cells, "cell_type": [extract_cell_type(c) for c in cells], "fold": folds.to_numpy()})
    fold_table.to_csv(TABLES / "a05_tissue_folds.csv", index=False)
    print("Cells per fold:", fold_table["fold"].value_counts().sort_index().to_dict(), "| types:", fold_table["cell_type"].nunique(), flush=True)
    enhancers = training["enhancer_id"].tolist()
    actual = openness.loc[enhancers, cells].to_numpy()
    results = {}
    for model in args.models:
        predicted = np.full_like(actual, np.nan, dtype=float)
        start = time.time()
        for fold in range(N_FOLDS):
            train_cells = [c for c, f in zip(cells, folds) if f != fold]
            test_cells = [c for c, f in zip(cells, folds) if f == fold]
            test_idx = [cells.index(c) for c in test_cells]

            # Reuse the tested section 7 loop: train on the fold's training cells, "profiles" = held-out cells.
            table = run_perd_model(training, rna[train_cells], openness[train_cells], rna[test_cells], model=model, n_jobs=n_jobs)
            predicted[:, test_idx] = table.loc[enhancers, test_cells].to_numpy()
            print(f"{model} fold {fold}: {len(test_cells)} held-out cells, {time.time() - start:.0f}s elapsed", flush=True)
        np.save(OUT / f"{model}_out_of_fold.npy", predicted.astype(np.float32))
        results[f"tissue_grouped_{model}"] = metrics(actual, predicted) | {"minutes": (time.time() - start) / 60}
    held_out = pd.read_parquet(PERD / "netopen_rf_predictions.parquet").set_index("enhancer_id").loc[enhancers]
    results["random_split_rf_57_cells"] = metrics(np.vstack(held_out["y_test"].to_numpy()), np.vstack(held_out["y_pred"].to_numpy()))
    (TABLES / "a05_tissue_split_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
