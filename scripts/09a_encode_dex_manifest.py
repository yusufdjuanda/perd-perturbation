"""Amendment 03 A: list the ENCODE A549 dexamethasone files to download (metadata only, no data).

For each experiment in the time-course table: DNase `read-depth normalized signal` bigWig
(GRCh38) and RNA `gene quantifications` tsv (GRCh38, annotation V29), one file per biological
replicate, with size, md5, and URL. Also records mapped-read counts from the experiments'
alignment-file quality metrics where ENCODE provides them. Writes data/encode_dex_manifest.json.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "encode_dex_manifest.json"
PORTAL = "https://www.encodeproject.org"
SERIES = [  # (time in hours, DNase experiment, RNA experiment), amendment 03 section A
    (0, "ENCSR136DNA", "ENCSR632DQP"),
    (0.5, "ENCSR406EMB", "ENCSR070IYV"),
    (1, "ENCSR384KCZ", "ENCSR656FIH"),
    (2, "ENCSR837VHE", "ENCSR924BHF"),
    (3, "ENCSR294XUZ", "ENCSR624RID"),
    (4, "ENCSR599WJC", "ENCSR326PTW"),
    (5, "ENCSR565WPR", "ENCSR543QRU"),
    (6, "ENCSR077EYC", "ENCSR255VBV"),
    (7, "ENCSR347CEH", "ENCSR385TFN"),
    (8, "ENCSR660OQE", "ENCSR546PPG"),
    (10, "ENCSR128IVG", "ENCSR224PTG"),
    (12, "ENCSR523FJT", "ENCSR154TDP"),
]


def get(path: str, params: dict | None = None) -> dict:
    url = PORTAL + path + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "perd-perturbation/0.1"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def search_files(experiment: str, **filters) -> list[dict]:
    params = {"type": "File", "dataset": f"/experiments/{experiment}/", "status": "released", "assembly": "GRCh38", "format": "json", "frame": "object", "limit": "all", **filters}
    return get("/search/", params).get("@graph", [])


def per_replicate(files: list[dict], label: str, problems: list[str]) -> list[dict]:
    """One file per biological replicate; files covering several replicates (pooled) are dropped."""
    by_rep: dict[int, list[dict]] = {}
    for f in files:
        reps = f.get("biological_replicates", [])
        if len(reps) == 1:
            by_rep.setdefault(reps[0], []).append(f)
    chosen = []
    for rep, candidates in sorted(by_rep.items()):
        if len(candidates) > 1:
            preferred = [c for c in candidates if c.get("preferred_default")]
            pick = preferred[0] if len(preferred) == 1 else sorted(candidates, key=lambda c: c.get("date_created", ""))[-1]
            problems.append(f"{label} replicate {rep}: {len(candidates)} candidates ({[c['accession'] for c in candidates]}), chose {pick['accession']}")
        else:
            pick = candidates[0]
        chosen.append(pick)
    return chosen


def mapped_reads(experiment: str) -> dict[str, dict]:
    """Mapped reads per biological replicate from alignment-file quality metrics (best effort)."""
    out = {}
    for f in search_files(experiment, output_type="alignments", file_format="bam"):
        reps = f.get("biological_replicates", [])
        if len(reps) != 1:
            continue
        metrics = {}
        for qm_id in f.get("quality_metrics", []):
            qm = get(qm_id, {"format": "json", "frame": "object"})
            for key in ("mapped", "uniquely_mapped_reads", "Uniquely mapped reads number", "mapped_reads"):
                if key in qm:
                    metrics[f"{qm.get('@type', ['?'])[0]}:{key}"] = qm[key]
        out[str(reps[0])] = {"alignment_file": f["accession"], **metrics}
    return out


def record(f: dict, time_h: float, experiment: str, assay: str, ext: str) -> dict:
    accession = f["accession"]
    return {
        "time_h": time_h,
        "assay": assay,
        "experiment": experiment,
        "file_accession": accession,
        "biological_replicate": f["biological_replicates"][0],
        "output_type": f.get("output_type"),
        "file_format": f.get("file_format"),
        "assembly": f.get("assembly"),
        "genome_annotation": f.get("genome_annotation"),
        "preferred_default": f.get("preferred_default"),
        "size_bytes": f.get("file_size"),
        "md5sum": f.get("md5sum"),
        "url": f"{PORTAL}/files/{accession}/@@download/{accession}.{ext}",
    }


def main() -> None:
    files, problems, depth = [], [], {}
    for time_h, dnase, rna in SERIES:
        d = per_replicate(search_files(dnase, output_type="read-depth normalized signal", file_format="bigWig"), f"DNase {dnase}", problems)
        r = per_replicate(search_files(rna, output_type="gene quantifications", file_format="tsv", genome_annotation="V29"), f"RNA {rna}", problems)
        files += [record(f, time_h, dnase, "DNase", "bigWig") for f in d] + [record(f, time_h, rna, "RNA", "tsv") for f in r]
        depth[dnase] = mapped_reads(dnase)
        depth[rna] = mapped_reads(rna)
        print(f"{time_h:>4} h  DNase {dnase}: {len(d)} replicates  RNA {rna}: {len(r)} replicates", flush=True)
    dnase_bytes = sum(f["size_bytes"] or 0 for f in files if f["assay"] == "DNase")
    rna_bytes = sum(f["size_bytes"] or 0 for f in files if f["assay"] == "RNA")
    payload = {
        "schema_version": 1,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source": PORTAL,
        "series": [{"time_h": t, "dnase_experiment": d, "rna_experiment": r} for t, d, r in SERIES],
        "files": files,
        "selection_notes": problems,
        "alignment_depth": depth,
        "totals": {"dnase_files": sum(f["assay"] == "DNase" for f in files), "dnase_bytes": dnase_bytes, "rna_files": sum(f["assay"] == "RNA" for f in files), "rna_bytes": rna_bytes},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"], indent=2))
    print("Selection notes:", json.dumps(problems, indent=2))
    print(json.dumps(depth, indent=1)[:4000])


if __name__ == "__main__":
    main()
