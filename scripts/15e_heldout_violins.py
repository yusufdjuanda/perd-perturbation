"""Post hoc (user request, 2026-10-07): per-enhancer and per-cell accuracy for paper-style violins.

(1) PERD-RF, random 110/57 split (../perd netopen_rf_predictions); (2) PERD-RF, tissue held out; (3) M1-RF (shared random
forest, 16b), tissue held out. (2) and (3) are reported on all 167 cells and on the same 57 test cells as (1); the 57 cell
IDs are recovered by exact matching of y_test against enhancer_openness.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
TABLES = ROOT / "results" / "tables"
SHARED = ROOT / "data" / "processed" / "shared_model"
SPLIT = ROOT / "data" / "processed" / "tissue_split"


def per_enhancer_r(a: np.ndarray, p: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    p = p - p.mean(axis=1, keepdims=True)
    den = np.sqrt((a**2).sum(axis=1) * (p**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * p).sum(axis=1) / den, np.nan)


def per_cell(a: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = per_enhancer_r(a.T, p.T)
    nspe = ((a - p) ** 2).sum(axis=0) / ((a - a.mean(axis=0)) ** 2).sum(axis=0)
    return r, nspe


def main() -> None:
    cells = list(pd.read_parquet(PERD / "rna_167_cells_symbol.parquet").columns)
    enhancers = pd.Index(pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"])
    actual = pd.read_parquet(PERD / "enhancer_openness.parquet").loc[enhancers, cells].to_numpy(dtype=np.float64)

    held = pd.read_parquet(PERD / "netopen_rf_predictions.parquet").set_index("enhancer_id").loc[enhancers]
    y_test = np.vstack(held["y_test"].to_numpy()).astype(np.float64)
    y_pred = np.vstack(held["y_pred"].to_numpy()).astype(np.float64)
    probe = np.argsort(-actual.std(axis=1))[:300]
    test_idx = []
    for j in range(y_test.shape[1]):
        hits = [c for c in range(len(cells)) if np.allclose(actual[probe, c], y_test[probe, j], atol=1e-5)]
        if len(hits) != 1:
            raise SystemExit(f"test column {j}: {len(hits)} matching cells; stop")
        test_idx.append(hits[0])
    assert np.allclose(actual[:, test_idx], y_test, atol=1e-5, equal_nan=True)
    print(f"recovered {len(test_idx)} test cells (unique, exact match on all enhancers)")

    perd_tissue = np.load(SPLIT / "rf_out_of_fold.npy").astype(np.float64)
    shared = np.full_like(actual, np.nan)
    for f in sorted(SHARED.glob("M1_rf_fold*.json")):
        meta = json.loads(f.read_text())
        block = np.load(SHARED / f"M1_rf_fold{meta['fold']}.npy")
        shared[:, [cells.index(c) for c in meta["test_cells"]]] = block
    if np.isnan(shared).all(axis=0).any():
        raise SystemExit("M1-RF folds missing; run 16b for all folds first")

    sets = {
        ("PERD-RF", "Random 110/57 split", "57 test cells"): (y_test, y_pred),
        ("PERD-RF", "Tissue held out", "57 test cells"): (actual[:, test_idx], perd_tissue[:, test_idx]),
        ("PERD-RF", "Tissue held out", "all 167 cells"): (actual, perd_tissue),
        ("Shared RF", "Tissue held out", "57 test cells"): (actual[:, test_idx], shared[:, test_idx]),
        ("Shared RF", "Tissue held out", "all 167 cells"): (actual, shared),
    }
    enh_rows, cell_rows, summary = [], [], []
    for (model, scheme, cellset), (a, p) in sets.items():
        r = per_enhancer_r(a, p)
        cr, cn = per_cell(a, p)
        enh_rows.append(pd.DataFrame({"model": model, "scheme": scheme, "cells": cellset, "cross_cell_r": r}))
        cell_rows.append(pd.DataFrame({"model": model, "scheme": scheme, "cells": cellset, "cross_enhancer_r": cr, "nspe": cn}))
        pooled = float(((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum())
        summary.append({"model": model, "scheme": scheme, "cells": cellset, "median_cross_cell_r": float(np.nanmedian(r)),
                        "frac_r_above_0.5": float(np.nanmean(r > 0.5)), "median_cross_enhancer_r": float(np.median(cr)), "pooled_nspe": pooled})
    pd.concat(enh_rows).to_csv(SHARED / "violin_per_enhancer.csv.gz", index=False)
    pd.concat(cell_rows).to_csv(TABLES / "heldout_violin_per_cell.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(TABLES / "heldout_violin_summary.csv", index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
