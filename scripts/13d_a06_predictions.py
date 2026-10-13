"""Amendment 06 D: frozen PERD-RF and PERD-ridge predictions for every K562 and T-cell RNA replicate (one pass).

Prediction side only. Writes data/processed/encode_{k562,tcell}/perd_predictions_{rf,ridge}.parquet.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation.perd_rf import run_perd_model

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    tables = {d: pd.read_parquet(PROCESSED / f"encode_{d}" / "rna_log2fpkm_replicates.parquet") for d in ("k562", "tcell")}
    genes = tables["k562"].index.union(tables["tcell"].index)
    profiles = pd.concat([t.reindex(genes).fillna(0.0) for t in tables.values()], axis=1)
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    encode_rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    summary = {"profiles": int(profiles.shape[1]), "genes": int(profiles.shape[0])}
    for model in ("rf", "ridge"):
        start = time.time()
        table = run_perd_model(training, encode_rna, openness, profiles, model=model, n_jobs=n_jobs)
        seconds = time.time() - start
        fitted = table.index[table["skip_reason"] == ""]
        chosen = np.random.default_rng(0).choice(fitted.to_numpy(), size=200, replace=False)
        rerun = run_perd_model(training[training["enhancer_id"].isin(chosen)], encode_rna, openness, profiles, model=model, n_jobs=n_jobs)
        identical = bool(np.array_equal(rerun[list(profiles.columns)].to_numpy(), table.loc[rerun.index, list(profiles.columns)].to_numpy()))
        meta = [c for c in ("n_genes_train", "n_genes_used", "skip_reason", "alpha") if c in table.columns]
        for dataset, t in tables.items():
            table[list(t.columns) + meta].to_parquet(PROCESSED / f"encode_{dataset}" / f"perd_predictions_{model}.parquet")
        summary[model] = {"fitted": len(fitted), "skipped": int((table["skip_reason"] != "").sum()), "minutes": seconds / 60, "determinism_identical": identical}
        print(model, summary[model], flush=True)
        if not identical:
            raise SystemExit(f"{model}: determinism check FAILED")
    (TABLES / "a06_prediction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
