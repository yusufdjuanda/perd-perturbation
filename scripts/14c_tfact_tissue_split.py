"""Amendment 07 D (secondary): held-out-tissue accuracy of PERD with TF activity features. ENCODE data only.

Same 5 tissue-grouped folds, models and metrics as amendment 05 E (11d); features = the variant's lists, with activity
rows from 14a (reference scale from all 167 cells, an openness-free transform; noted deviation).
Usage: python scripts/14c_tfact_tissue_split.py tfadd|tfrep
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
OUT = ROOT / "data" / "processed" / "tissue_split"


def load_split_module():
    spec = importlib.util.spec_from_file_location("split", ROOT / "scripts" / "11d_tissue_grouped_split.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    variant = sys.argv[1]
    if variant not in ("tfadd", "tfrep"):
        raise SystemExit("usage: 14c_tfact_tissue_split.py tfadd|tfrep")
    split = load_split_module()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"]
    targets = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")["target_genes"]
    activity = pd.read_parquet(ROOT / "data" / "processed" / "tf_activity" / "activity_encode167.parquet")
    training = tf_activity.feature_lists(frozen_training, binding, targets, set(activity.index), variant)
    rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    features = tf_activity.with_activity(rna, activity)
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    cells = list(rna.columns)
    folds = split.fold_assignment(cells)
    enhancers = training["enhancer_id"].tolist()
    actual = openness.loc[enhancers, cells].to_numpy()
    results = {}
    for model in ("ridge", "rf"):
        predicted = np.full_like(actual, np.nan, dtype=float)
        start = time.time()
        for fold in range(split.N_FOLDS):
            train_cells = [c for c, f in zip(cells, folds) if f != fold]
            test_cells = [c for c, f in zip(cells, folds) if f == fold]
            test_idx = [cells.index(c) for c in test_cells]
            table = run_perd_model(training, features[train_cells], openness[train_cells], features[test_cells], model=model, n_jobs=n_jobs)
            predicted[:, test_idx] = table.loc[enhancers, test_cells].to_numpy()
            print(f"{variant} {model} fold {fold}: {len(test_cells)} held-out cells, {time.time() - start:.0f}s", flush=True)
        np.save(OUT / f"{model}_{variant}_out_of_fold.npy", predicted.astype(np.float32))
        results[f"tissue_grouped_{model}"] = split.metrics(actual, predicted) | {"minutes": (time.time() - start) / 60}
    results["frozen"] = json.loads((TABLES / "a05_tissue_split_metrics.json").read_text())
    (TABLES / f"a07_tissue_split_metrics_{variant}.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
