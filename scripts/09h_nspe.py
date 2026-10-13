"""POST HOC, descriptive: PERD's normalized squared prediction error (NSPE) on our test data.

NSPE = sum((O - P)^2) / sum((O - mean(O))^2); 0 = perfect, 1 = no better than predicting the mean.
Defined in execution.md (2026-10-04) before computing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
MIN_MEAN = 1.0


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    ok = ~np.isnan(observed) & ~np.isnan(predicted)
    o, p = observed[ok], predicted[ok]
    sse = np.sum((o - p) ** 2)
    sst = np.sum((o - o.mean()) ** 2)
    r = float(np.corrcoef(o, p)[0, 1])
    return {
        "n": int(ok.sum()),
        "nspe": float(sse / sst),
        "pearson": r,
        "nspe_best_linear_rescale": float(1 - r**2),
        "rmse": float(np.sqrt(np.mean((o - p) ** 2))),
        "mse": float(np.mean((o - p) ** 2)),
        "mse_zero_predictor": float(np.mean(o**2)),
        "mse_mean_predictor": float(np.mean((o - o.mean()) ** 2)),
    }


def main() -> None:
    rows = []
    files = pd.DataFrame(json.loads((ROOT / "data" / "encode_dex_manifest.json").read_text())["files"]).sort_values(["time_h", "biological_replicate"])
    dnase = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "DNase"].groupby("time_h")}
    rna = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "RNA"].groupby("time_h")}
    openness = pd.read_parquet(PROCESSED / "encode_dex" / "enhancer_openness_replicates.parquet")
    for model in ("rf", "ridge"):
        pred = pd.read_parquet(PROCESSED / "encode_dex" / f"perd_predictions_{model}.parquet")
        fitted = pred.index[pred["skip_reason"] == ""]
        measured = openness.loc[fitted]
        control_m, control_p = measured[dnase[0]].mean(axis=1), pred.loc[fitted, rna[0]].mean(axis=1)
        for t in sorted(dnase):
            m_abs, p_abs = measured[dnase[t]].mean(axis=1), pred.loc[fitted, rna[t]].mean(axis=1)
            rows.append({"dataset": "ENCODE dexamethasone", "model": model, "quantity": "absolute openness", "time_h": t, **metrics(m_abs.to_numpy(), p_abs.to_numpy())})
            if t == 0:
                continue
            evaluable = ((m_abs + control_m) / 2 >= MIN_MEAN) & m_abs.notna() & control_m.notna()
            m_change, p_change = (m_abs - control_m)[evaluable], (p_abs - control_p)[evaluable]
            rows.append({"dataset": "ENCODE dexamethasone", "model": model, "quantity": "change vs 0 h", "time_h": t, **metrics(m_change.to_numpy(), p_change.to_numpy())})

    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "nspe_test_data.csv", index=False)
    with pd.option_context("display.width", 200):
        summary = table.groupby(["dataset", "model", "quantity"])[["nspe", "pearson", "nspe_best_linear_rescale", "rmse", "mse", "mse_zero_predictor"]].mean()
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
