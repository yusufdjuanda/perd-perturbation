"""Amendment 10 E4 evaluation (written before scoring): held-out predictions of E4, B0 and E4-noBase vs frozen PERD.

Per condition, on enhancers that are evaluable and fitted by frozen PERD: extreme-prediction z (K = 1%, strata by
control-openness decile x frozen n_genes tercile), change NSPE and Pearson on F1, switch-on AUROC.
Units: mean z and median NSPE over their conditions. P1: E4 - frozen PERD-ridge mean z (16 units, SK-N-SH excluded);
P2: median E4 NSPE < 1 and E4 - B0 NSPE paired bootstrap; P3: E4 - B0 mean z. Bootstrap over units, 10,000 draws,
seed 42, 97.5%.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from perd_perturbation import plate_eval, variant

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
OUT = PROC / "pair_model"
MIN_MEAN, CLOSED, OPEN = 1.0, 1.0, 2.0
MODELS = ("E4", "B0", "noBase")
EXCLUDED_FROM_FROZEN = {"SK-N-SH retinoic acid"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_module("pipeline", ROOT / "scripts" / "11b_perd_pipeline_dex.py")
nspe_mod = load_module("nspe", ROOT / "scripts" / "09h_nspe.py")


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = label.sum(), (~label).sum()
    if pos == 0 or neg == 0:
        return np.nan
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def score(cond: dict, predictor: str, pred: np.ndarray, measured: np.ndarray, control: np.ndarray, stratum: np.ndarray, f1: np.ndarray, treated: np.ndarray) -> dict:
    k = plate_eval.k_values(len(pred))[0.01]
    ext = plate_eval.extreme_test_components(pred, measured, stratum, {0.01: k}, n_perm=2000, seed=42)[0.01]
    o, p = measured[f1], pred[f1]
    m = nspe_mod.metrics(o, p)
    row = {"dataset": cond["dataset"], "unit": cond["unit"], "condition": cond["name"], "predictor": predictor, "n": len(pred), "z": ext["z"],
           "nspe_F1": float(m["nspe"]), "pearson_F1": float(np.corrcoef(o, p)[0, 1]) if np.std(p) > 0 else np.nan}
    if cond.get("threshold") is not None:
        closed = control < CLOSED
        event = (treated >= OPEN) & (measured > cond["threshold"])
        row["switch_on_auroc"] = auroc(pred[closed], event[closed])
        row["switch_on_events"] = int(event[closed].sum())
    return row


def boot_units(diff: pd.Series, n: int = 10_000) -> list[float]:
    rng = np.random.default_rng(42)
    vals = diff.to_numpy()
    draws = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return [float(x) for x in np.percentile(draws, [1.25, 98.75])]


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    conditions = json.loads((OUT / "conditions.json").read_text())
    tasks = []
    cache = {}
    for cond in conditions:
        ds = cond["dataset"]
        if ds not in cache:
            frozen = {m: variant.read_predictions(PROC / ds, m, variant="") for m in ("rf", "ridge")}
            fitted = frozen["rf"].index[frozen["rf"]["skip_reason"] == ""]
            f1 = pipeline.candidate_sets(fitted, set(pd.read_parquet(PROC / ds / "rna_log2fpkm_replicates.parquet").index))[0]["F1"]
            cache[ds] = {"open": pd.read_parquet(PROC / ds / "enhancer_openness_replicates.parquet"), "frozen": frozen, "fitted": fitted, "f1": f1}
        c = cache[ds]
        control = c["open"].loc[c["fitted"], cond["acc_control"]].mean(axis=1)
        treated = c["open"].loc[c["fitted"], cond["acc_treated"]].mean(axis=1)
        ev = control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)
        idx = ev[ev].index
        measured = (treated - control).loc[idx].to_numpy()
        stratum = plate_eval.strata(control.loc[idx].to_numpy(), c["frozen"]["rf"]["n_genes_used"].loc[idx].to_numpy())
        f1 = idx.isin(c["f1"])
        preds = {f"frozen_{m}": (c["frozen"][m].loc[idx, cond["rna_treated"]].mean(axis=1) - c["frozen"][m].loc[idx, cond["rna_control"]].mean(axis=1)).to_numpy() for m in ("rf", "ridge")}
        for model in MODELS:
            path = OUT / f"pred_{model}_{cond['unit'].replace(' ', '_')}.parquet"
            if path.exists():
                preds[model] = pd.read_parquet(path).loc[idx, cond["name"]].to_numpy()
        for name, p in preds.items():
            tasks.append((cond, name, p, measured, control.loc[idx].to_numpy(), stratum, f1, treated.loc[idx].to_numpy()))
    rows = Parallel(n_jobs=n_jobs)(delayed(score)(*t) for t in tasks)
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "a10_e4_per_condition.csv", index=False)

    unit = table.groupby(["unit", "dataset", "predictor"]).agg(z=("z", "mean"), nspe_F1=("nspe_F1", "median"), pearson_F1=("pearson_F1", "median"), switch_on_auroc=("switch_on_auroc", "median")).reset_index()
    unit.to_csv(TABLES / "a10_e4_per_unit.csv", index=False)
    wide_z = unit.pivot(index="unit", columns="predictor", values="z")
    wide_n = unit.pivot(index="unit", columns="predictor", values="nspe_F1")
    comp = wide_z.drop(index=[u for u in EXCLUDED_FROM_FROZEN if u in wide_z.index])
    summary = {"units": int(len(wide_z)), "units_vs_frozen": int(len(comp))}
    if "E4" in wide_z:
        d = (comp["E4"] - comp["frozen_ridge"]).dropna()
        summary["P1_E4_minus_frozen_ridge_z"] = {"mean": float(d.mean()), "ci": boot_units(d), "n_units": len(d), "improves": bool(boot_units(d)[0] > 0)}
        d_rf = (comp["E4"] - comp["frozen_rf"]).dropna()
        summary["E4_minus_frozen_rf_z"] = {"mean": float(d_rf.mean()), "ci": boot_units(d_rf)}
        summary["P2_E4_median_nspe"] = float(wide_n["E4"].median())
        if "B0" in wide_n:
            dn = (wide_n["E4"] - wide_n["B0"]).dropna()
            ci = boot_units(dn)
            summary["P2_E4_minus_B0_nspe"] = {"mean": float(dn.mean()), "ci": ci, "B0_median_nspe": float(wide_n["B0"].median())}
            summary["P2_predicts_effect_size"] = bool(summary["P2_E4_median_nspe"] < 1.0 and ci[1] < 0)
            dz = (wide_z["E4"] - wide_z["B0"]).dropna()
            summary["P3_E4_minus_B0_z"] = {"mean": float(dz.mean()), "ci": boot_units(dz), "change_features_add": bool(boot_units(dz)[0] > 0)}
        summary["median_nspe_by_predictor"] = {k: float(v) for k, v in wide_n.median().items()}
        summary["mean_z_by_predictor"] = {k: float(v) for k, v in wide_z.mean().items()}
    (TABLES / "a10_e4_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(wide_z.round(2).to_string())
        print(wide_n.round(3).to_string())
        print(unit.groupby(["dataset", "predictor"])[["z", "nspe_F1", "pearson_F1", "switch_on_auroc"]].mean().round(3).to_string())
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
