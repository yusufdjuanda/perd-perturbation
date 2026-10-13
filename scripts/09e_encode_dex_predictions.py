"""Amendment 03 section D: frozen PERD-RF and PERD-ridge predictions for the ENCODE dexamethasone RNA replicates.

Prediction side only; no DNase value is read. Details fixed in execution.md (2026-10-04) before running.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation import encode_dex, evaluation
from perd_perturbation.perd_rf import run_perd_model


ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
OUT = ROOT / "data" / "processed" / "encode_dex"
TABLES = ROOT / "results" / "tables"
MANIFEST = ROOT / "data" / "encode_dex_manifest.json"
DETERMINISM_N = 200
DETERMINISM_SEED = 0
QS = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    files = pd.DataFrame(json.loads(MANIFEST.read_text())["files"])
    rna = files[files["assay"] == "RNA"].sort_values(["time_h", "biological_replicate"])
    reps = {t: list(zip(g["biological_replicate"], g["file_accession"])) for t, g in rna.groupby("time_h")}
    profiles = pd.read_parquet(OUT / "rna_log2fpkm_replicates.parquet")[rna["file_accession"].tolist()]
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    encode_rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    summary: dict = {"profiles": int(profiles.shape[1]), "genes": int(profiles.shape[0])}

    for model in ("rf", "ridge"):
        start = time.time()
        table = run_perd_model(training, encode_rna, openness, profiles, model=model, n_jobs=n_jobs)
        seconds = time.time() - start
        table.to_parquet(OUT / f"perd_predictions_{model}.parquet")
        fitted = table.index[table["skip_reason"] == ""]
        chosen = np.random.default_rng(DETERMINISM_SEED).choice(fitted.to_numpy(), size=DETERMINISM_N, replace=False)
        rerun = run_perd_model(training[training["enhancer_id"].isin(chosen)], encode_rna, openness, profiles, model=model, n_jobs=n_jobs)
        cols = list(profiles.columns)
        identical = bool(np.array_equal(rerun[cols].to_numpy(), table.loc[rerun.index, cols].to_numpy()))
        t = table.loc[fitted]

        def mean_of(accessions):
            return t[accessions].mean(axis=1)

        control = mean_of([a for _, a in reps[0]])
        per_time, band_values, band_rows = {}, [], []
        for time_h, replicate_list in reps.items():
            by_number = dict(replicate_list)
            half_a, half_b = encode_dex.replicate_halves(list(by_number))
            diff = mean_of([by_number[n] for n in half_a]) - mean_of([by_number[n] for n in half_b])
            band_values.append(diff.abs())
            band_rows.append({"time_h": float(time_h), "p95": float(diff.abs().quantile(0.95)), "p99": float(diff.abs().quantile(0.99)), "halves": [half_a, half_b]})
            if time_h == 0:
                continue
            change = mean_of([a for _, a in replicate_list]) - control
            ctrl = dict(reps[0])
            c_a, c_b = encode_dex.replicate_halves(list(ctrl))
            change_a = mean_of([by_number[n] for n in half_a]) - mean_of([ctrl[n] for n in c_a])
            change_b = mean_of([by_number[n] for n in half_b]) - mean_of([ctrl[n] for n in c_b])
            per_time[float(time_h)] = {
                "quantiles": {str(q): round(float(v), 4) for q, v in change.quantile(QS).items()},
                "fraction_exactly_zero": float((change == 0).mean()),
                "split_half_spearman_of_predicted_change": evaluation.correlation(change_a, change_b),
            }
        pooled = pd.concat(band_values)
        summary[model] = {
            "fitted": len(fitted),
            "skipped_by_reason": {k: int(v) for k, v in table["skip_reason"].value_counts().items() if k},
            "n_genes_used_quantiles": t["n_genes_used"].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(),
            "fit_minutes": seconds / 60,
            "determinism_identical": identical,
            "noise_band_pooled_p95": float(pooled.quantile(0.95)),
            "noise_band_pooled_p99": float(pooled.quantile(0.99)),
            "noise_band_by_time": band_rows,
            "predicted_change_by_time": per_time,
        }
        print(f"{model}: fitted {len(fitted):,} in {seconds / 60:.1f} min; determinism {identical}", flush=True)
        if not identical:
            raise SystemExit(f"{model}: determinism check FAILED")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "encode_dex_prediction_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    for model in ("rf", "ridge"):
        s = summary[model]
        print(model, "band p95/p99", round(s["noise_band_pooled_p95"], 3), round(s["noise_band_pooled_p99"], 3))
        for time_h, v in s["predicted_change_by_time"].items():
            print(f"  {time_h:>4} h  p5 {v['quantiles']['0.05']:+.3f}  p50 {v['quantiles']['0.5']:+.3f}  p95 {v['quantiles']['0.95']:+.3f}  zero {v['fraction_exactly_zero']:.4f}  split-half r {v['split_half_spearman_of_predicted_change']:.3f}")


if __name__ == "__main__":
    main()
