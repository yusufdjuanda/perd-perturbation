"""Amendment 07 A-B: CollecTRI regulon, TF activity (ULM) for training and test samples, and the PERD-blind gate.

No openness value is read. Writes data/processed/tf_activity/activity_*.parquet, data/collectri_manifest.json,
results/tables/a07_activity_gate.json and a07_activity_descriptives.csv. Exits non-zero if the gate fails.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation import tf_activity

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
OUT = PROCESSED / "tf_activity"
RAW = ROOT / "data" / "raw" / "collectri"
TABLES = ROOT / "results" / "tables"
URL = "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&loops=yes&format=tsv"
TCELL_TFS = ["NFATC1", "NFATC2", "RELA", "NFKB1", "IRF4", "JUN", "FOS"]


def download() -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "collectri.tsv"
    if not path.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "perd-perturbation/0.1"})
        with urllib.request.urlopen(request, timeout=600) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    raw = pd.read_csv(io.BytesIO(data), sep="\t")
    manifest = {"url": URL, "retrieved_at": datetime.now(timezone.utc).isoformat(), "path": str(path.relative_to(ROOT)), "md5": hashlib.md5(data).hexdigest(), "bytes": len(data), "rows": int(len(raw))}
    (ROOT / "data" / "collectri_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return raw


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = download()
    network = tf_activity.network_from_collectri(raw)
    encode = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    mean, sd = tf_activity.reference_scale(encode)
    weights = tf_activity.weight_matrix(network, mean.index)
    report = {"collectri_rows": int(len(raw)), "network_edges": int(len(network)), "network_tfs": int(network["source"].nunique()), "reference_genes": int(len(mean)), "scored_tfs": int(weights.shape[1])}

    tables = {
        "encode167": encode,
        "a549_dex": pd.read_parquet(PROCESSED / "encode_dex" / "rna_log2fpkm_replicates.parquet"),
        "k562": pd.read_parquet(PROCESSED / "encode_k562" / "rna_log2fpkm_replicates.parquet"),
        "tcell": pd.read_parquet(PROCESSED / "encode_tcell" / "rna_log2fpkm_replicates.parquet"),
    }
    activities = {}
    for name, table in tables.items():
        act = tf_activity.activity(table, mean, sd, weights)
        act.to_parquet(OUT / f"activity_{name}.parquet")
        activities[name] = act
        report[f"{name}_samples"] = int(act.shape[1])
        report[f"{name}_reference_genes_present"] = int(table.index.isin(mean.index).sum())

    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet")
    binding_tfs = {tf for tfs in binding["tf_genes"] for tf in tfs}
    report["binding_tfs"] = len(binding_tfs)
    report["binding_tfs_scored"] = len(binding_tfs & set(weights.columns))
    report["binding_tfs_not_scored"] = sorted(binding_tfs - set(weights.columns))

    # ---- Gate B (A549): NR3C1 activity change > 0 at >= 9 of 11 time points ----
    files = pd.DataFrame(json.loads((ROOT / "data" / "encode_dex_manifest.json").read_text())["files"])
    rna = files[files["assay"] == "RNA"]
    by_time = {float(t): g["file_accession"].tolist() for t, g in rna.groupby("time_h")}
    a549 = activities["a549_dex"]
    control = a549[by_time[0.0]].mean(axis=1)
    changes = pd.DataFrame({t: a549[acc].mean(axis=1) - control for t, acc in by_time.items() if t != 0.0})
    nr3c1 = changes.loc["NR3C1"]
    positive = int((nr3c1 > 0).sum())
    mean_change = changes.mean(axis=1).sort_values(ascending=False)
    enc_nr3c1 = activities["encode167"].loc["NR3C1"]
    a549_levels = a549.loc["NR3C1"]
    gate = {
        "nr3c1_change_by_time": {str(t): float(v) for t, v in nr3c1.items()},
        "time_points_positive": positive,
        "passes": positive >= 9,
        "nr3c1_rank_by_mean_change": int(list(mean_change.index).index("NR3C1") + 1),
        "n_tfs_ranked": int(len(mean_change)),
        "top10_tfs_by_mean_change": {k: float(v) for k, v in mean_change.head(10).items()},
        "encode167_nr3c1": {"mean": float(enc_nr3c1.mean()), "sd": float(enc_nr3c1.std()), "min": float(enc_nr3c1.min()), "max": float(enc_nr3c1.max())},
        "a549_nr3c1_level_by_time": {str(t): float(a549_levels[acc].mean()) for t, acc in by_time.items()},
        "a549_replicates_above_encode_max": int((a549_levels > enc_nr3c1.max()).sum()),
        "a549_replicates": int(a549_levels.size),
    }
    report["gate"] = gate

    # ---- descriptives: T cells and K562 ----
    rows = []
    samples_t = pd.read_csv(PROCESSED / "encode_tcell" / "samples.csv")
    rna_t = samples_t[samples_t["assay"] == "total RNA-seq"]
    for pair, g in rna_t.groupby("pair"):
        act = activities["tcell"]
        diff = act[g[g["state"] == "activated"]["file_accession"]].mean(axis=1) - act[g[g["state"] == "resting"]["file_accession"]].mean(axis=1)
        for tf in TCELL_TFS:
            rows.append({"dataset": "tcell", "condition": pair, "tf": tf, "activity_change": float(diff.get(tf, np.nan)), "rank": int(diff.rank(ascending=False).get(tf, -1)) if tf in diff else None})
    samples_k = pd.read_csv(PROCESSED / "encode_k562" / "samples.csv")
    rna_k = samples_k[samples_k["assay"] == "total RNA-seq"]
    act = activities["k562"]
    for (drug, t), g in rna_k[rna_k["drug"] != "DMSO"].groupby(["drug", "time_h"]):
        dmso = rna_k[(rna_k["drug"] == "DMSO") & (rna_k["time_h"] == t)]["file_accession"]
        if dmso.empty:
            continue
        diff = (act[g["file_accession"]].mean(axis=1) - act[dmso].mean(axis=1)).sort_values(ascending=False)
        for rank, (tf, value) in enumerate(diff.head(10).items(), start=1):
            rows.append({"dataset": "k562", "condition": f"{drug} {t:g}h", "tf": tf, "activity_change": float(value), "rank": rank})
    pd.DataFrame(rows).to_csv(TABLES / "a07_activity_descriptives.csv", index=False)
    changes.to_csv(TABLES / "a07_a549_activity_change_by_time.csv")

    (TABLES / "a07_activity_gate.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "binding_tfs_not_scored"}, indent=2))
    print("binding TFs without a score:", report["binding_tfs_not_scored"])
    if not gate["passes"]:
        raise SystemExit("Gate B FAILED: NR3C1 activity does not rise at >= 9/11 time points. Stop and report (amendment 07 B).")


if __name__ == "__main__":
    main()
