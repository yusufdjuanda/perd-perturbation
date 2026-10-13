"""Amendment 06 A3: select the K562 and T-cell files from the saved ENCODE metadata (no download).

Writes data/encode_k562_manifest.json and data/encode_tcell_manifest.json; prints the selection and sizes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PORTAL = "https://www.encodeproject.org"
PRIMARY = ["Bortezomib", "Chaetocin", "GSK J4", "JQ1"]
SENSITIVITY = ["AR-42", "C646", "Crizotinib", "Galeterone", "MB-3", "Panobinostat", "Vorinostat"]
TCELL_PAIRS = [
    ("CD4 T", "ENCDO394DKX", "activated CD4-positive, alpha-beta T cell", "CD4-positive, alpha-beta T cell"),
    ("CD8 T", "ENCDO781BMZ", "activated CD8-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell"),
    ("CD4 memory T", "ENCDO791UJG", "activated CD4-positive, alpha-beta memory T cell", "CD4-positive, alpha-beta memory T cell"),
    ("naive CD4 T", "ENCDO791UJG", "activated naive CD4-positive, alpha-beta T cell", "naive thymus-derived CD4-positive, alpha-beta T cell"),
    ("naive CD8 T", "ENCDO166AGR", "activated naive CD8-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell"),
]
ACTIVATION = "anti-CD3 and anti-CD28 coated beads 36 hour"


def parse_k562(treatment: str):
    m = re.match(r"^(.*?)\s+(?:([\d.]+)\s*(μM|nM|percent)\s+)?([\d.]+)\s*hour$", treatment)
    if not m:
        return None
    drug = "DMSO" if m.group(1).strip().upper() == "DMSO" else m.group(1).strip()
    dose = f"{m.group(2)} {m.group(3)}" if m.group(2) else ""
    return drug, dose, float(m.group(4))


def file_record(f: pd.Series, extra: dict) -> dict:
    ext = {"bam": "bam", "bigWig": "bigWig", "tsv": "tsv"}[f["file_format"]]
    return {
        **extra,
        "experiment": f["experiment"],
        "file_accession": f["file_accession"],
        "output_type": f["output_type"],
        "file_format": f["file_format"],
        "biological_replicate": f["biological_replicates"],
        "genome_annotation": f.get("genome_annotation"),
        "size_bytes": int(f["size_bytes"]) if pd.notna(f["size_bytes"]) else None,
        "md5sum": f["md5sum"],
        "url": f"{PORTAL}/files/{f['file_accession']}/@@download/{f['file_accession']}.{ext}",
    }


def pick(files: pd.DataFrame, experiment: str, output_type: str, file_format: str) -> pd.DataFrame:
    """One file per single biological replicate; preferred default first, else the first listed."""
    sub = files[(files["experiment"] == experiment) & (files["output_type"] == output_type) & (files["file_format"] == file_format)]
    sub = sub[~sub["biological_replicates"].astype(str).str.contains(",")]
    sub = sub.sort_values("preferred_default", ascending=False, key=lambda s: s.fillna(False).astype(bool))
    return sub.drop_duplicates("biological_replicates")


def main() -> None:
    experiments = pd.read_csv(TABLES / "encode_k562_tcell_experiments.csv")
    files = pd.read_csv(TABLES / "encode_k562_tcell_files.csv")
    files = files[files["assembly"] == "GRCh38"]
    now = datetime.now(timezone.utc).isoformat()

    # ---- K562 ----
    k562 = experiments[(experiments["dataset"] == "K562") & experiments["assay"].isin(["ATAC-seq", "total RNA-seq"])].copy()
    parsed = k562["treatments"].map(parse_k562)
    k562 = k562[parsed.notna()].assign(drug=parsed[parsed.notna()].map(lambda p: p[0]), dose=parsed[parsed.notna()].map(lambda p: p[1]), time_h=parsed[parsed.notna()].map(lambda p: p[2]))
    k562 = k562[k562["drug"].isin(PRIMARY + SENSITIVITY + ["DMSO"])]
    k562_records = []
    for row in k562.itertuples():
        output_type, fmt = ("alignments", "bam") if row.assay == "ATAC-seq" else ("gene quantifications", "tsv")
        group = "primary" if row.drug in PRIMARY else "sensitivity" if row.drug in SENSITIVITY else "control"
        for _, f in pick(files, row.experiment, output_type, fmt).iterrows():
            k562_records.append(file_record(f, {"assay": row.assay, "drug": row.drug, "dose": row.dose, "time_h": row.time_h, "group": group}))
    k562_manifest = pd.DataFrame(k562_records)

    # ---- T cells ----
    tcell_records, exclusions = [], []
    tcell = experiments[experiments["dataset"] == "T cell"]
    for pair, donor, activated_term, resting_term in TCELL_PAIRS:
        found = {}
        for state, term, treatment in (("activated", activated_term, ACTIVATION), ("resting", resting_term, "none")):
            for assay, output_type in (("DNase-seq", "read-depth normalized signal"), ("total RNA-seq", "gene quantifications")):
                sel = tcell[(tcell["cell"] == term) & (tcell["treatments"] == treatment) & (tcell["assay"] == assay) & tcell["donors"].fillna("").str.split(",").map(lambda d: donor in d)]
                fmt = "bigWig" if assay == "DNase-seq" else "tsv"
                chosen = [f for exp in sel["experiment"] for _, f in pick(files, exp, output_type, fmt).iterrows()]
                found[(state, assay)] = len(chosen)
                tcell_records += [file_record(f, {"pair": pair, "donor": donor, "state": state, "cell_term": term, "assay": assay}) for f in chosen]
        missing = [f"{s} {a}" for (s, a), n in found.items() if n == 0]
        if missing:
            exclusions.append({"pair": pair, "donor": donor, "missing": missing})
    tcell_manifest = pd.DataFrame(tcell_records)
    if exclusions:
        excluded_pairs = {e["pair"] for e in exclusions}
        tcell_manifest = tcell_manifest[~tcell_manifest["pair"].isin(excluded_pairs)]

    for name, manifest, extra in (("k562", k562_manifest, {}), ("tcell", tcell_manifest, {"excluded_pairs": exclusions})):
        payload = {"generated_at": now, "source": "results/tables/encode_k562_tcell_files.csv (ENCODE portal metadata, 2026-10-05)", **extra, "files": manifest.to_dict("records"), "total_bytes": int(manifest["size_bytes"].fillna(0).sum())}
        (ROOT / "data" / f"encode_{name}_manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")

    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print("K562:", k562_manifest.groupby(["assay", "group"]).agg(files=("file_accession", "count"), gb=("size_bytes", lambda s: s.sum() / 1e9)).round(2).to_string())
        print(k562_manifest.groupby(["assay", "drug"])["time_h"].apply(lambda t: sorted(set(t))).to_string())
        print("\nT cells:", tcell_manifest.groupby(["pair", "state", "assay"]).agg(files=("file_accession", "count"), gb=("size_bytes", lambda s: s.sum() / 1e9)).round(2).to_string())
        print("Excluded pairs:", json.dumps(exclusions, indent=1))
        total = (k562_manifest["size_bytes"].sum() + tcell_manifest["size_bytes"].sum()) / 1e9
        print(f"\nTotal download: {total:.1f} GB (K562 {k562_manifest['size_bytes'].sum() / 1e9:.1f}, T cell {tcell_manifest['size_bytes'].sum() / 1e9:.1f})")


if __name__ == "__main__":
    main()
