"""Amendment 06 A/B: download, md5-check, and reduce the K562 and T-cell files; delete large inputs after use.

- K562 ATAC BAM (paired-end, ENCODE-filtered) -> primary mapped reads with MAPQ >= 30 on chr1-22/X outside
  the blacklist -> each mate's 5' end shifted (+4 / -5) = Tn5 cut site -> base-pair cut sites + 200 bp bins.
- T-cell DNase bigWig -> mean signal per enhancer grid bin and promoter window (amendment 03 B2 code).
- RNA gene-quantification tsv -> kept.
Resumable: status JSON per file in data/interim/a06_status/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation import atac, encode_dex, runs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
STATUS = ROOT / "data" / "interim" / "a06_status"
GRID = PROCESSED / "encode_dex" / "grid.parquet"
PROMOTERS = PROCESSED / "encode_dex" / "promoters.parquet"
BLACKLIST = RAW / "reference" / "hg38-blacklist.v2.bed.gz"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> float:
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        try:
            start = time.time()
            request = urllib.request.Request(url, headers={"User-Agent": "perd-perturbation/0.1"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            partial.replace(destination)
            return time.time() - start
        except Exception:
            if attempt == 4:
                raise
            time.sleep(15 * 2**attempt)
    raise RuntimeError("unreachable")


def process(dataset: str, record: dict) -> dict:
    accession = record["file_accession"]
    qc = {"dataset": dataset, "file_accession": accession, "assay": record["assay"], "output_type": record["output_type"]}
    started = time.time()
    raw_dir = RAW / f"encode_{dataset}" / ("rna" if record["file_format"] == "tsv" else "tmp")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{accession}.{record['file_format']}"
    out = PROCESSED / f"encode_{dataset}"
    try:
        if not path.exists():
            qc["download_seconds"] = download(record["url"], path)
        qc["bytes"] = path.stat().st_size
        if md5(path) != record["md5sum"]:
            raise ValueError("md5 mismatch")
        if record["file_format"] == "bam":
            bed = path.with_suffix(".bed")
            runs.sh(f"samtools view -u -q 30 -F 0x904 {path} | bedtools bamtobed -i stdin > {bed}")
            reads = atac.read_bed_reads(bed)
            blacklist = pd.read_csv(BLACKLIST, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
            kept, counts = atac.filter_reads(reads, blacklist)
            cuts = atac.tn5_cut_sites(kept)
            meta = {"reads_final": counts["reads_final"], "reads_in": counts["reads_in"]}
            (out / "atac_cuts").mkdir(parents=True, exist_ok=True)
            (out / "atac_bins").mkdir(parents=True, exist_ok=True)
            atac.save_cut_sites(out / "atac_cuts" / f"{accession}.npz", cuts, meta)
            atac.save_bins(out / "atac_bins" / f"{accession}.npz", atac.bin_counts(cuts), meta)
            qc.update({k: int(v) for k, v in counts.items()})
            bed.unlink()
            path.unlink()
        elif record["file_format"] == "bigWig":
            import pyBigWig

            grid, promoters = pd.read_parquet(GRID), pd.read_parquet(PROMOTERS)
            with pyBigWig.open(str(path)) as bw:
                bins = encode_dex.bigwig_window_means(bw, encode_dex.grid_windows(grid))
                promoter_means = encode_dex.bigwig_window_means(bw, promoters[["chrom", "start", "end"]])
            (out / "bins").mkdir(parents=True, exist_ok=True)
            (out / "promoters").mkdir(parents=True, exist_ok=True)
            np.save(out / "bins" / f"{accession}.npy", bins.astype(np.float32))
            np.save(out / "promoters" / f"{accession}.npy", promoter_means.astype(np.float32))
            qc["bins_nonzero_fraction"] = float((bins > 0).mean())
            path.unlink()
        status = "done"
    except Exception as error:
        status = f"failed: {type(error).__name__}: {error}"
        qc["traceback"] = traceback.format_exc()
    qc["status"] = status
    qc["total_seconds"] = time.time() - started
    qc["finished_at"] = datetime.now(timezone.utc).isoformat()
    (STATUS / f"{accession}.json").write_text(json.dumps(qc, indent=2) + "\n")
    return qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    STATUS.mkdir(parents=True, exist_ok=True)
    todo = []
    for dataset in ("tcell", "k562"):
        for record in json.loads((ROOT / "data" / f"encode_{dataset}_manifest.json").read_text())["files"]:
            status_file = STATUS / f"{record['file_accession']}.json"
            if status_file.exists():
                status = json.loads(status_file.read_text())["status"]
                if status == "done" or not args.retry_failed:
                    continue
            todo.append((dataset, record))
    # Largest files first so the pool's tail is short.
    todo.sort(key=lambda item: -(item[1].get("size_bytes") or 0))
    print(f"To process: {len(todo)} files, {sum((r.get('size_bytes') or 0) for _, r in todo) / 1e9:.1f} GB", flush=True)
    start = time.time()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, dataset, record) for dataset, record in todo]
        for future in as_completed(futures):
            qc = future.result()
            done += qc["status"] == "done"
            failed += qc["status"] != "done"
            print(f"[{time.time() - start:6.0f}s] {qc['dataset']} {qc['file_accession']} {qc['assay']} {qc['status'][:80]} ({qc['total_seconds']:.0f}s) done={done} failed={failed}", flush=True)
    print(f"Finished in {time.time() - start:.0f}s: {done} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
