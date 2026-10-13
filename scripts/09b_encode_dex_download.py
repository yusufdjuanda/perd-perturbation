"""Amendment 03 A/B2: download the ENCODE dexamethasone files, verify md5, extract DNase bins, delete bigWigs.

DNase: per replicate, mean signal over every 200 bp bin strictly contained in a PERD training
enhancer (`bins/<ACC>.npy`, order in `grid.parquet`) and over protein-coding promoter windows
(`promoters/<ACC>.npy`, order in `promoters.parquet`); the bigWig is then deleted.
RNA: the gene-quantification tsv is kept in data/raw/encode_dex/rna/.
Resumable: status files in data/interim/encode_dex_status/.
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

from perd_perturbation import atac, encode_dex


ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
MANIFEST = ROOT / "data" / "encode_dex_manifest.json"
RAW = ROOT / "data" / "raw" / "encode_dex"
OUT = ROOT / "data" / "processed" / "encode_dex"
STATUS = ROOT / "data" / "interim" / "encode_dex_status"
GTF = ROOT / "data" / "raw" / "reference" / "gencode.v44.annotation.gtf.gz"
PROMOTER_FLANK = 500
VERIFY_BINS = 3


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


def build_windows() -> None:
    """Grid of contained bins and promoter windows, written once."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "grid.parquet").exists() and (OUT / "promoters.parquet").exists():
        return
    genehancer = pd.read_parquet(PERD / "genehancer_enhancers.parquet").set_index("enhancer_id")
    training = genehancer.loc[pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"]].reset_index()
    grid = atac.contained_grid_bins(training[training["chrom"].isin(atac.CANONICAL)])
    grid.to_parquet(OUT / "grid.parquet")
    tss = atac.tss_table(GTF)
    tss = tss[(tss["gene_type"] == "protein_coding") & tss["chrom"].isin(atac.CANONICAL)].drop_duplicates("gene_id")
    promoters = tss.assign(start=tss["pos"] - PROMOTER_FLANK, end=tss["pos"] + PROMOTER_FLANK + 1)[["gene_id", "gene_name", "chrom", "start", "end", "strand", "pos"]]
    promoters.reset_index(drop=True).to_parquet(OUT / "promoters.parquet")
    print(f"Grid bins: {len(grid):,}; promoter windows: {len(promoters):,}", flush=True)


def process(record: dict, verify: bool) -> dict:
    import pyBigWig

    accession, assay = record["file_accession"], record["assay"]
    qc = {"file_accession": accession, "assay": assay, "time_h": record["time_h"], "biological_replicate": record["biological_replicate"]}
    started = time.time()
    ext = "bigWig" if assay == "DNase" else "tsv"
    target_dir = RAW / ("dnase" if assay == "DNase" else "rna")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{accession}.{ext}"
    try:
        if not path.exists():
            qc["download_seconds"] = download(record["url"], path)
        qc["bytes"] = path.stat().st_size
        qc["md5_observed"] = md5(path)
        if qc["md5_observed"] != record["md5sum"]:
            raise ValueError(f"md5 mismatch: {qc['md5_observed']} != {record['md5sum']}")
        if assay == "DNase":
            grid = pd.read_parquet(OUT / "grid.parquet")
            promoters = pd.read_parquet(OUT / "promoters.parquet")
            windows = encode_dex.grid_windows(grid)
            with pyBigWig.open(str(path)) as bw:
                bins = encode_dex.bigwig_window_means(bw, windows)
                promoter_means = encode_dex.bigwig_window_means(bw, promoters[["chrom", "start", "end"]])
                if verify:
                    checks = []
                    for i in np.random.default_rng(42).choice(np.flatnonzero(bins > 0), VERIFY_BINS, replace=False):
                        w = windows.iloc[i]
                        second = encode_dex.interval_mean(bw, w["chrom"], int(w["start"]), int(w["end"]))
                        checks.append({"chrom": w["chrom"], "start": int(w["start"]), "method1": float(bins[i]), "method2": second})
                    qc["verification"] = checks
                    if not all(abs(c["method1"] - c["method2"]) <= 1e-5 * max(1.0, abs(c["method2"])) for c in checks):
                        raise ValueError(f"bin verification failed: {checks}")
            (OUT / "bins").mkdir(exist_ok=True)
            (OUT / "promoters").mkdir(exist_ok=True)
            np.save(OUT / "bins" / f"{accession}.npy", bins.astype(np.float32))
            np.save(OUT / "promoters" / f"{accession}.npy", promoter_means.astype(np.float32))
            qc["bins_nonzero_fraction"] = float((bins > 0).mean())
            qc["bin_mean"] = float(bins.mean())
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
    build_windows()
    files = json.loads(MANIFEST.read_text())["files"]
    todo = []
    for record in files:
        status_file = STATUS / f"{record['file_accession']}.json"
        if status_file.exists():
            status = json.loads(status_file.read_text())["status"]
            if status == "done" or not args.retry_failed:
                continue
        todo.append(record)
    first_dnase = next((r["file_accession"] for r in todo if r["assay"] == "DNase"), None)
    print(f"Files: {len(files)}; to process: {len(todo)}", flush=True)
    start = time.time()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, r, r["file_accession"] == first_dnase) for r in todo]
        for future in as_completed(futures):
            qc = future.result()
            done += qc["status"] == "done"
            failed += qc["status"] != "done"
            print(f"[{time.time() - start:6.0f}s] {qc['file_accession']} {qc['assay']} {qc['time_h']}h rep{qc['biological_replicate']} {qc['status'][:90]} ({qc['total_seconds']:.0f}s) done={done} failed={failed}", flush=True)
    print(f"Finished in {time.time() - start:.0f}s: {done} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
