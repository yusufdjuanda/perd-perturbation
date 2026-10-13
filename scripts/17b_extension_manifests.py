"""Amendment 09 B: download manifests for the HCT116 auxin-degron and SK-N-SH retinoic-acid series (metadata only).

DNase `read-depth normalized signal` bigWig and RNA `gene quantifications` tsv, GRCh38, one file per biological
replicate (preferred default first). Writes data/encode_hct116_manifest.json and data/encode_sknsh_manifest.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
PORTAL = "https://www.encodeproject.org"
SKIP = {"purpose:expression", "purpose:tagging"}


def target_of(mods: str) -> str:
    labels = [x for x in str(mods).split(";") if x and x not in SKIP and not x.startswith("tag:") and x != "nan"]
    return labels[0] if labels else ""


def pick(files: pd.DataFrame) -> pd.DataFrame:
    files = files[~files["biological_replicates"].astype(str).str.contains(",")]
    files = files.sort_values("preferred_default", ascending=False, key=lambda s: s.fillna(False).astype(bool))
    return files.drop_duplicates(["experiment", "biological_replicates"])


def record(f: pd.Series, extra: dict) -> dict:
    ext = {"bigWig": "bigWig", "tsv": "tsv"}[f["file_format"]]
    return {**extra, "experiment": f["experiment"], "file_accession": f["file_accession"], "output_type": f["output_type"], "file_format": f["file_format"],
            "biological_replicate": str(f["biological_replicates"]), "size_bytes": int(f["file_size"]), "md5sum": f["md5sum"],
            "url": f"{PORTAL}/files/{f['file_accession']}/@@download/{f['file_accession']}.{ext}"}


def main() -> None:
    summary = pd.read_csv(TABLES / "encode_extension_perturbation_summary.csv")
    files = pd.read_csv(TABLES / "encode_extension_perturbation_files.csv")
    files = files[files["assembly"] == "GRCh38"]
    now = datetime.now(timezone.utc).isoformat()
    for name, series, rna_assays in (("hct116", "HCT116 + auxin (5-Ph-IAA)", ["total RNA-seq"]), ("sknsh", "SK-N-SH + retinoic acid", ["polyA plus RNA-seq"])):
        exps = summary[summary["series"] == series].copy()
        exps["unit"] = exps["genetic_modification"].map(target_of) if name == "hct116" else "retinoic acid"
        if name == "hct116":
            exps = exps[exps["unit"] != ""]
        exps = exps[exps["assay"].isin(["DNase-seq", *rna_assays])]
        records = []
        for e in exps.itertuples():
            out_type, fmt = ("read-depth normalized signal", "bigWig") if e.assay == "DNase-seq" else ("gene quantifications", "tsv")
            sub = pick(files[(files["experiment"] == e.experiment) & (files["output_type"] == out_type) & (files["file_format"] == fmt)])
            for _, f in sub.iterrows():
                records.append(record(f, {"assay": e.assay, "unit": e.unit, "state": e.state, "treatments": e.treatments}))
        manifest = pd.DataFrame(records)
        units = manifest.groupby(["unit", "assay", "state"]).agg(files=("file_accession", "count"), gb=("size_bytes", lambda s: s.sum() / 1e9)).round(2)
        complete = [u for u in manifest["unit"].unique() if all(((manifest["unit"] == u) & (manifest["assay"] == a) & (manifest["state"] == s)).any()
                                                               for a in ("DNase-seq", rna_assays[0]) for s in ("treated", "untreated"))]
        payload = {"generated_at": now, "source": "results/tables/encode_extension_perturbation_files.csv (ENCODE metadata, 2026-10-06)", "series": series,
                   "complete_units": complete, "files": manifest.to_dict("records"), "total_bytes": int(manifest["size_bytes"].sum())}
        (ROOT / "data" / f"encode_{name}_manifest.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(f"== {name}: {len(manifest)} files, {manifest['size_bytes'].sum() / 1e9:.1f} GB; complete units: {complete}")
            print(units.to_string())


if __name__ == "__main__":
    main()
