"""Post hoc (user request, 2026-10-08): NSPE per held-out cell for PERD-RF input variants (frozen, + TF activity,
+ motifs, + both, shuffled control), tissue-grouped out-of-fold predictions on all 167 ENCODE cells. Descriptive only;
the prespecified comparisons (amendments 07, 09, 10 M) used cross-cell correlation and are unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
TABLES = ROOT / "results" / "tables"
SPLIT = ROOT / "data" / "processed" / "tissue_split"
VARIANTS = {"rf": "Frozen PERD-RF", "rf_tfshuf": "+ shuffled-target activity", "rf_tfadd": "+ TF activity",
            "rf_tfmotif": "+ motif activity", "rf_tfboth": "+ both"}


def main() -> None:
    cells = list(pd.read_parquet(PERD / "rna_167_cells_symbol.parquet").columns)
    enhancers = pd.Index(pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"])
    actual = pd.read_parquet(PERD / "enhancer_openness.parquet").loc[enhancers, cells].to_numpy(dtype=np.float64)
    rows, summary = [], []
    for key, label in VARIANTS.items():
        pred = np.load(SPLIT / f"{key}_out_of_fold.npy").astype(np.float64)
        per_cell = ((actual - pred) ** 2).sum(axis=0) / ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
        rows.append(pd.DataFrame({"variant": label, "cell": cells, "nspe": per_cell}))
        pooled = float(((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum())
        summary.append({"variant": label, "median_cell_nspe": float(np.median(per_cell)), "pooled_nspe": pooled})
    pd.concat(rows).to_csv(TABLES / "inputs_heldout_nspe_per_cell.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(TABLES / "inputs_heldout_nspe_summary.csv", index=False)
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
