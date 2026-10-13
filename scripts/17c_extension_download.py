"""Amendment 09 B: download, md5-check and reduce the HCT116 and SK-N-SH files (same worker as amendment 06, 13b)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("worker", ROOT / "scripts" / "13b_k562_tcell_download.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def run(dataset: str, record: dict) -> dict:
    """Top-level wrapper so worker processes can pickle the task."""
    return worker.process(dataset, record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    worker.STATUS.mkdir(parents=True, exist_ok=True)
    todo = []
    for dataset in ("sknsh", "hct116"):
        for record in json.loads((ROOT / "data" / f"encode_{dataset}_manifest.json").read_text())["files"]:
            status = worker.STATUS / f"{record['file_accession']}.json"
            if status.exists() and json.loads(status.read_text())["status"] == "done":
                continue
            todo.append((dataset, record))
    todo.sort(key=lambda item: -(item[1].get("size_bytes") or 0))
    print(f"To process: {len(todo)} files, {sum(r.get('size_bytes') or 0 for _, r in todo) / 1e9:.1f} GB", flush=True)
    start, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run, d, r) for d, r in todo]):
            qc = future.result()
            done += qc["status"] == "done"
            failed += qc["status"] != "done"
            print(f"[{time.time() - start:6.0f}s] {qc['dataset']} {qc['file_accession']} {qc['assay']} {qc['status'][:80]} done={done} failed={failed}", flush=True)
    print(f"Finished in {time.time() - start:.0f}s: {done} done, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
