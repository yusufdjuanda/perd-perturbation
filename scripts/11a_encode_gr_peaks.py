"""Amendment 05 D: download ENCODE A549 NR3C1 (glucocorticoid receptor) ChIP-seq peaks, dexamethasone 100 nM, 0.5-12 h.

GRCh38 conservative IDR thresholded peaks (bed narrowPeak), preferred default where several exist;
md5-verified; manifest in data/encode_gr_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "encode_gr"
MANIFEST = ROOT / "data" / "encode_gr_manifest.json"
PORTAL = "https://www.encodeproject.org"
SERIES = {0.5: "ENCSR773NQB", 1: "ENCSR919OXR", 2: "ENCSR720DXT", 3: "ENCSR385RUW", 4: "ENCSR660RYY", 5: "ENCSR686PFM", 6: "ENCSR424JOE", 7: "ENCSR952GOL", 8: "ENCSR691GRA", 10: "ENCSR255CTA", 12: "ENCSR287FWE"}


def get(path: str, params: dict) -> dict:
    url = PORTAL + path + "?" + urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "perd-perturbation/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for time_h, experiment in SERIES.items():
        files = get("/search/", {"type": "File", "dataset": f"/experiments/{experiment}/", "output_type": "conservative IDR thresholded peaks", "assembly": "GRCh38", "file_format": "bed", "status": "released", "format": "json", "frame": "object", "limit": "all"})["@graph"]
        if not files:
            raise SystemExit(f"No conservative IDR peaks for {experiment}")
        preferred = [f for f in files if f.get("preferred_default")]
        pick = preferred[0] if preferred else sorted(files, key=lambda f: f.get("date_created", ""))[-1]
        url = f"{PORTAL}/files/{pick['accession']}/@@download/{pick['accession']}.bed.gz"
        path = OUT / f"{pick['accession']}.bed.gz"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "perd-perturbation/0.1"}), timeout=300) as response:
            path.write_bytes(response.read())
        observed = hashlib.md5(path.read_bytes()).hexdigest()
        if observed != pick["md5sum"]:
            raise SystemExit(f"md5 mismatch for {pick['accession']}")
        records.append({"time_h": time_h, "experiment": experiment, "file_accession": pick["accession"], "candidates": len(files), "preferred_default": pick.get("preferred_default"), "size_bytes": path.stat().st_size, "md5sum": observed, "url": url})
        print(time_h, experiment, pick["accession"], len(files), path.stat().st_size, flush=True)
    MANIFEST.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(), "files": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
