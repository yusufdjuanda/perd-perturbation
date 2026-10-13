"""Post hoc (user-approved, 2026-10-08): every model on every perturbation, scored by NSPE.

Conditions and enhancers exactly as 20b (40 conditions, 16 perturbations; evaluable and fitted by frozen PERD; NSPE on
PERD's candidate set F1). PERD-RF versions (frozen, tfadd, tfmotif, tfboth): before, after and change NSPE. Pair-model
predictors (E4, B0, noBase): change NSPE. Descriptive only.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation import variant

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
PAIR = PROC / "pair_model"
MIN_MEAN = 1.0
PERD_VERSIONS = {"frozen_rf": "", "tfadd_rf": "tfadd", "tfmotif_rf": "tfmotif", "tfboth_rf": "tfboth"}
PAIR_MODELS = ("E4", "B0", "noBase")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_module("pipeline", ROOT / "scripts" / "11b_perd_pipeline_dex.py")


def nspe(o: np.ndarray, p: np.ndarray) -> float:
    return float(((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum())


def main() -> None:
    conditions = json.loads((PAIR / "conditions.json").read_text())
    cache, rows = {}, []
    for cond in conditions:
        ds = cond["dataset"]
        if ds not in cache:
            preds = {k: variant.read_predictions(PROC / ds, "rf", variant=v) for k, v in PERD_VERSIONS.items()}
            fitted = preds["frozen_rf"].index[preds["frozen_rf"]["skip_reason"] == ""]
            f1 = pipeline.candidate_sets(fitted, set(pd.read_parquet(PROC / ds / "rna_log2fpkm_replicates.parquet").index))[0]["F1"]
            cache[ds] = {"open": pd.read_parquet(PROC / ds / "enhancer_openness_replicates.parquet"), "preds": preds, "fitted": fitted, "f1": f1}
        c = cache[ds]
        control = c["open"].loc[c["fitted"], cond["acc_control"]].mean(axis=1)
        treated = c["open"].loc[c["fitted"], cond["acc_treated"]].mean(axis=1)
        ev = control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)
        idx = ev[ev].index
        f1 = idx[idx.isin(c["f1"])]
        base = {"dataset": ds, "unit": cond["unit"], "condition": cond["name"], "n_F1": len(f1)}
        m_before, m_after = control.loc[f1].to_numpy(), treated.loc[f1].to_numpy()
        m_change = m_after - m_before
        for key, table in c["preds"].items():
            p_before = table.loc[f1, cond["rna_control"]].mean(axis=1).to_numpy()
            p_after = table.loc[f1, cond["rna_treated"]].mean(axis=1).to_numpy()
            rows.append({**base, "predictor": key, "nspe_before": nspe(m_before, p_before), "nspe_after": nspe(m_after, p_after),
                         "nspe_change": nspe(m_change, p_after - p_before)})
        for model in PAIR_MODELS:
            path = PAIR / f"pred_{model}_{cond['unit'].replace(' ', '_')}.parquet"
            p = pd.read_parquet(path).loc[f1, cond["name"]].to_numpy()
            rows.append({**base, "predictor": model, "nspe_before": np.nan, "nspe_after": np.nan, "nspe_change": nspe(m_change, p)})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "a11_all_models_per_condition.csv", index=False)
    metrics = ["nspe_before", "nspe_after", "nspe_change"]
    unit = table.groupby(["dataset", "unit", "predictor"])[metrics].median().reset_index()
    unit.to_csv(TABLES / "a11_all_models_per_unit.csv", index=False)
    by_ds = table.groupby(["dataset", "predictor"])[metrics].median().reset_index()
    overall = unit.groupby("predictor")[metrics].median().reset_index()
    by_ds.to_csv(TABLES / "a11_all_models_by_dataset.csv", index=False)
    overall.to_csv(TABLES / "a11_all_models_overall.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(by_ds.round(3).to_string())
        print(overall.round(3).to_string())


if __name__ == "__main__":
    main()
