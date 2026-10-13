"""POST HOC (definitions in execution.md, 2026-10-06): uncertainty for the TF-activity gain on held-out ENCODE
tissues (bootstrap over tissue types), and the A549 NR3C1 mRNA vs activity table for the figure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
SPLIT = ROOT / "data" / "processed" / "tissue_split"
TABLES = ROOT / "results" / "tables"
VARIANTS = ("tfadd", "tfrep")


def per_cell(actual: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = actual - actual.mean(axis=0)
    p = pred - pred.mean(axis=0)
    r = (a * p).sum(axis=0) / np.sqrt((a * a).sum(axis=0) * (p * p).sum(axis=0))
    nspe = ((actual - pred) ** 2).sum(axis=0) / (a * a).sum(axis=0)
    return r, nspe


def cross_cell(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    a = actual - actual.mean(axis=1, keepdims=True)
    p = pred - pred.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * p).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (p * p).sum(axis=1))


def main() -> None:
    cells = list(pd.read_parquet(PERD / "rna_167_cells_symbol.parquet").columns)
    enhancers = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"]
    actual = pd.read_parquet(PERD / "enhancer_openness.parquet").loc[enhancers, cells].to_numpy(dtype=np.float32)
    folds = pd.read_csv(TABLES / "a05_tissue_folds.csv").set_index("cell").loc[cells]
    types = folds["cell_type"].to_numpy()
    tissue_types = np.unique(types)
    cells_of = {t: np.flatnonzero(types == t) for t in tissue_types}
    pred = {(v, m): np.load(SPLIT / f"{m}{'_' + v if v else ''}_out_of_fold.npy") for v in ("", *VARIANTS) for m in ("rf", "ridge")}

    rows, report = [], {}
    metrics = {k: per_cell(actual, p) for k, p in pred.items()}
    for (v, m), (r, n) in metrics.items():
        for j, c in enumerate(cells):
            rows.append({"version": v or "frozen", "model": m, "cell": c, "cell_type": types[j], "cross_enhancer_pearson": float(r[j]), "nspe": float(n[j])})
    pd.DataFrame(rows).to_csv(TABLES / "posthoc_tfact_heldout_per_cell.csv", index=False)

    rng = np.random.default_rng(42)
    draws = [rng.choice(tissue_types, len(tissue_types), replace=True) for _ in range(10_000)]
    for v in VARIANTS:
        for m in ("rf", "ridge"):
            out = {}
            for k, label in ((0, "cross_enhancer_pearson"), (1, "nspe")):
                diff = metrics[(v, m)][k] - metrics[("", m)][k]
                per_type = {t: diff[cells_of[t]] for t in tissue_types}
                boot = np.array([np.concatenate([per_type[t] for t in pick]).mean() for pick in draws])
                low, high = np.percentile(boot, [1.25, 98.75])
                out[label] = {"mean_diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high), "cells_improved": int((diff > 0).sum() if k == 0 else (diff < 0).sum()), "n_cells": len(cells)}
            report[f"{v}_{m}_per_cell"] = out

    rng2 = np.random.default_rng(42)
    small = [rng2.choice(tissue_types, len(tissue_types), replace=True) for _ in range(500)]
    observed = {k: cross_cell(actual, p) for k, p in pred.items()}
    for v in VARIANTS:
        for m in ("rf", "ridge"):
            base, var = observed[("", m)], observed[(v, m)]
            obs = {"median_r_frozen": float(np.nanmedian(base)), "median_r_variant": float(np.nanmedian(var)),
                   "frac_above_0.5_frozen": float(np.nanmean(base > 0.5)), "frac_above_0.5_variant": float(np.nanmean(var > 0.5))}
            d_med, d_frac = [], []
            for pick in small:
                idx = np.concatenate([cells_of[t] for t in pick])
                b = cross_cell(actual[:, idx], pred[("", m)][:, idx])
                w = cross_cell(actual[:, idx], pred[(v, m)][:, idx])
                d_med.append(np.nanmedian(w) - np.nanmedian(b))
                d_frac.append(np.nanmean(w > 0.5) - np.nanmean(b > 0.5))
            obs["median_r_diff_ci"] = [float(x) for x in np.percentile(d_med, [1.25, 98.75])]
            obs["frac_above_0.5_diff_ci"] = [float(x) for x in np.percentile(d_frac, [1.25, 98.75])]
            report[f"{v}_{m}_cross_cell"] = obs
            print(v, m, json.dumps(obs), flush=True)
    (TABLES / "posthoc_tfact_heldout_bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")

    files = pd.DataFrame(json.loads((ROOT / "data" / "encode_dex_manifest.json").read_text())["files"])
    rna_files = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "RNA"].groupby("time_h")}
    mrna = pd.read_parquet(ROOT / "data" / "processed" / "encode_dex" / "rna_log2fpkm_replicates.parquet").loc["NR3C1"]
    act = pd.read_parquet(ROOT / "data" / "processed" / "tf_activity" / "activity_a549_dex.parquet").loc["NR3C1"]
    gr = pd.DataFrame({"time_h": list(rna_files), "nr3c1_mrna": [float(mrna[a].mean()) for a in rna_files.values()], "nr3c1_activity": [float(act[a].mean()) for a in rna_files.values()],
                       "nr3c1_mrna_sd": [float(mrna[a].std()) for a in rna_files.values()], "nr3c1_activity_sd": [float(act[a].std()) for a in rna_files.values()]})
    gr["mrna_change"] = gr["nr3c1_mrna"] - gr.loc[gr["time_h"] == 0, "nr3c1_mrna"].iloc[0]
    gr["activity_change"] = gr["nr3c1_activity"] - gr.loc[gr["time_h"] == 0, "nr3c1_activity"].iloc[0]
    gr.to_csv(TABLES / "a07_nr3c1_mrna_vs_activity.csv", index=False)
    print(json.dumps({k: v for k, v in report.items() if k.endswith("per_cell")}, indent=1))
    print(gr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
