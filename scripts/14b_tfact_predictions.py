"""Amendment 07 C: PERD-RF and PERD-ridge with TF activity features (variant tfadd or tfrep), fit on the 167 ENCODE cells.

Two prediction passes, exactly as frozen PERD: A549 dexamethasone RNA replicates (as 09e) and K562 + T-cell RNA
replicates (as 13d: RNA union of genes, missing = 0). Activity rows (act:TF) come from 14a. No openness of a test
sample is read. Usage: python scripts/14b_tfact_predictions.py tfadd|tfrep
"""

from __future__ import annotations

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
PROCESSED = ROOT / "data" / "processed"
ACT = PROCESSED / "tf_activity"
TABLES = ROOT / "results" / "tables"
DETERMINISM_N = 200


def predict(name: str, training: pd.DataFrame, train_features: pd.DataFrame, openness: pd.DataFrame, profiles: pd.DataFrame, n_jobs: int) -> dict:
    tables, info = {}, {}
    for model in ("rf", "ridge"):
        start = time.time()
        table = run_perd_model(training, train_features, openness, profiles, model=model, n_jobs=n_jobs)
        minutes = (time.time() - start) / 60
        fitted = table.index[table["skip_reason"] == ""]
        chosen = np.random.default_rng(0).choice(fitted.to_numpy(), size=DETERMINISM_N, replace=False)
        rerun = run_perd_model(training[training["enhancer_id"].isin(chosen)], train_features, openness, profiles, model=model, n_jobs=n_jobs)
        cols = list(profiles.columns)
        identical = bool(np.array_equal(rerun[cols].to_numpy(), table.loc[rerun.index, cols].to_numpy()))
        info[model] = {"fitted": int(len(fitted)), "skipped": int((table["skip_reason"] != "").sum()), "minutes": minutes, "determinism_identical": identical}
        print(name, model, info[model], flush=True)
        if not identical:
            raise SystemExit(f"{name} {model}: determinism check FAILED")
        tables[model] = table
    return {"tables": tables, "info": info}


def main() -> None:
    variant = sys.argv[1]
    if variant not in ("tfadd", "tfrep"):
        raise SystemExit("usage: 14b_tfact_predictions.py tfadd|tfrep")
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    encode_rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    frozen_training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"]
    targets = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")["target_genes"]
    act = {name: pd.read_parquet(ACT / f"activity_{name}.parquet") for name in ("encode167", "a549_dex", "k562", "tcell")}
    training = tf_activity.feature_lists(frozen_training, binding, targets, set(act["encode167"].index), variant)
    train_features = tf_activity.with_activity(encode_rna, act["encode167"])
    summary = {"variant": variant, "n_activity_features": training["n_activity"].describe().to_dict(), "enhancers_with_activity": int((training["n_activity"] > 0).sum()),
               "n_features": training["usable_genes"].map(len).describe().to_dict(), "frozen_n_features": frozen_training["usable_genes"].map(len).describe().to_dict()}
    print(json.dumps(summary, indent=1, default=float), flush=True)

    # ---- pass 1: A549 dexamethasone (as 09e) ----
    files = pd.DataFrame(json.loads((ROOT / "data" / "encode_dex_manifest.json").read_text())["files"])
    rna = files[files["assay"] == "RNA"].sort_values(["time_h", "biological_replicate"])
    a549 = pd.read_parquet(PROCESSED / "encode_dex" / "rna_log2fpkm_replicates.parquet")[rna["file_accession"].tolist()]
    result = predict("a549", training, train_features, openness, tf_activity.with_activity(a549, act["a549_dex"]), n_jobs)
    for model, table in result["tables"].items():
        table.to_parquet(PROCESSED / "encode_dex" / f"perd_predictions_{variant}_{model}.parquet")
    summary["a549"] = result["info"]

    # ---- pass 2: K562 + T cells (as 13d) ----
    rna_tables = {d: pd.read_parquet(PROCESSED / f"encode_{d}" / "rna_log2fpkm_replicates.parquet") for d in ("k562", "tcell")}
    genes = rna_tables["k562"].index.union(rna_tables["tcell"].index)
    profiles = pd.concat([t.reindex(genes).fillna(0.0) for t in rna_tables.values()], axis=1)
    activity = pd.concat([act["k562"], act["tcell"]], axis=1)
    result = predict("k562_tcell", training, train_features, openness, tf_activity.with_activity(profiles, activity), n_jobs)
    for model, table in result["tables"].items():
        meta = [c for c in ("n_genes_train", "n_genes_used", "skip_reason", "alpha") if c in table.columns]
        for dataset, t in rna_tables.items():
            table[list(t.columns) + meta].to_parquet(PROCESSED / f"encode_{dataset}" / f"perd_predictions_{variant}_{model}.parquet")
    summary["k562_tcell"] = result["info"]
    (TABLES / f"a07_prediction_summary_{variant}.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
