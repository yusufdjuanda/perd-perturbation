"""Amendment 07 A, method check (no openness read): our ULM vs decoupleR's ULM on identical inputs.

Same standardized expression (z against the ENCODE 167 reference) and the same network; decoupler's ulm should
reproduce our activity scores. Runs in a throwaway venv (data/interim/dc_venv), not the pixi environment. Writes results/tables/a07_decoupler_crosscheck.json.

Setup and run (decoupler is not in the pixi environment):
    pixi run python -m venv data/interim/dc_venv
    data/interim/dc_venv/bin/pip install decoupler pandas pyarrow numpy
    data/interim/dc_venv/bin/python scripts/14f_decoupler_crosscheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from perd_perturbation import tf_activity  # noqa: E402

import decoupler as dc  # noqa: E402

PERD = ROOT.parent / "perd" / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"


def run_ulm(z_samples_by_genes: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    """decoupler ulm, version 2.x API (dc.mt.ulm) or 1.x (dc.run_ulm). Returns samples x TFs estimates."""
    if hasattr(dc, "mt"):
        result = dc.mt.ulm(data=z_samples_by_genes, net=net, tmin=tf_activity.MIN_TARGETS, verbose=False)
        return result[0] if isinstance(result, tuple) else result
    estimate, _ = dc.run_ulm(mat=z_samples_by_genes, net=net, source="source", target="target", weight="weight", min_n=tf_activity.MIN_TARGETS, verbose=False)
    return estimate


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "raw" / "collectri" / "collectri.tsv", sep="\t")
    network = tf_activity.network_from_collectri(raw)
    encode = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    mean, sd = tf_activity.reference_scale(encode)
    a549 = pd.read_parquet(PROCESSED / "encode_dex" / "rna_log2fpkm_replicates.parquet")
    samples = {"a549_dex": a549, "encode167": encode.iloc[:, :20]}
    ours_tables = {name: pd.read_parquet(PROCESSED / "tf_activity" / f"activity_{name}.parquet") for name in samples}
    report = {"decoupler_version": getattr(dc, "__version__", "unknown")}
    for name, table in samples.items():
        z = tf_activity.standardize(table, mean, sd)
        theirs = run_ulm(z.T, network).T  # TFs x samples
        ours = ours_tables[name][z.columns]
        common = ours.index.intersection(theirs.index)
        a, b = ours.loc[common, z.columns].to_numpy(), theirs.loc[common, z.columns].to_numpy(dtype=float)
        report[name] = {
            "samples": int(z.shape[1]), "tfs_ours": int(ours.shape[0]), "tfs_decoupler": int(theirs.shape[0]), "tfs_common": int(len(common)),
            "max_abs_difference": float(np.nanmax(np.abs(a - b))), "pearson_all_values": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
            "nr3c1_ours_first5": [round(float(v), 4) for v in ours.loc["NR3C1"].iloc[:5]], "nr3c1_decoupler_first5": [round(float(v), 4) for v in theirs.loc["NR3C1"].iloc[:5]],
        }
    report["identical_within_1e-6"] = all(report[n]["max_abs_difference"] < 1e-6 and report[n]["tfs_common"] == report[n]["tfs_ours"] for n in samples)
    (TABLES / "a07_decoupler_crosscheck.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
