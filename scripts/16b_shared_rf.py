"""Post hoc (user request, 2026-10-07): amendment 08 M1 with a random forest instead of gradient boosting (M1-RF).

Same features, target, 20,000 training enhancers and tissue-grouped folds as 16a; only the learner changes.
Writes one out-of-fold block per fold: data/processed/shared_model/M1_rf_fold{k}.npy (enhancers x that fold's cells).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from perd_perturbation import shared_model, tf_activity

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "data" / "processed" / "shared_model"
N_TRAIN_ENHANCERS = 20_000


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load_module("split", ROOT / "scripts" / "11d_tissue_grouped_split.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    cells = list(rna.columns)
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    enhancers = pd.Index(training["enhancer_id"])
    actual = pd.read_parquet(PERD / "enhancer_openness.parquet").loc[enhancers, cells].to_numpy(dtype=np.float32)
    targets = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")["target_genes"]
    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"]
    activity = pd.read_parquet(ROOT / "data" / "processed" / "tf_activity" / "activity_encode167.parquet")
    mean, sd = tf_activity.reference_scale(rna)
    z = tf_activity.standardize(rna, mean, sd)
    static = shared_model.build_static(enhancers, targets, binding, mean.index, set(activity.index))
    print(f"static built: {len(enhancers):,} enhancers, {static.n_features} features; {time.time() - start:.0f}s", flush=True)

    folds = pd.read_csv(TABLES / "a05_tissue_folds.csv").set_index("cell").loc[cells]
    fold_of = folds["fold"].to_numpy()
    sub = np.sort(np.random.default_rng(42).choice(len(enhancers), size=N_TRAIN_ENHANCERS, replace=False))

    for fold in [int(f) for f in args.folds.split(",")]:
        t0 = time.time()
        train = [i for i in range(len(cells)) if fold_of[i] != fold]
        test = [i for i in range(len(cells)) if fold_of[i] == fold]
        prior_mean = actual[:, train].mean(axis=1)
        prior_sd = actual[:, train].std(axis=1)

        def feats(i: int) -> np.ndarray:
            return shared_model.cell_features(static, prior_mean, prior_sd, z[cells[i]], activity[cells[i]])

        x = np.vstack([feats(i)[sub] for i in train]).astype(np.float32)
        y = np.concatenate([actual[sub, i] - prior_mean[sub] for i in train])
        rf = RandomForestRegressor(n_estimators=500, max_features=x.shape[1] // 3, min_samples_leaf=100, n_jobs=n_jobs, random_state=42).fit(x, y)
        fit_min = (time.time() - t0) / 60
        del x, y
        block = np.column_stack([prior_mean + rf.predict(feats(i).astype(np.float32)) for i in test]).astype(np.float32)
        np.save(OUT / f"M1_rf_fold{fold}.npy", block)
        (OUT / f"M1_rf_fold{fold}.json").write_text(json.dumps({"fold": fold, "train_cells": len(train), "test_cells": [cells[i] for i in test],
                                                                 "fit_minutes": fit_min, "total_minutes": (time.time() - t0) / 60}) + "\n")
        print(f"fold {fold}: {len(train)} train / {len(test)} test cells, fit {fit_min:.1f} min, total {(time.time() - t0) / 60:.1f} min", flush=True)
        del rf


if __name__ == "__main__":
    main()
