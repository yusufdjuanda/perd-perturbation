"""Per-run processing steps shared by the pilot scripts and the plate worker.

Each function shells out to the bio tools (salmon, bowtie2, samtools, bedtools,
lbzip2), which must be on PATH (run under `pixi run -e bio`).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np


def sh(command: str) -> str:
    return subprocess.run(["bash", "-o", "pipefail", "-c", command], check=True, capture_output=True, text=True).stdout


def decompress(fastq_bz2: Path, destination: Path, threads: int = 4) -> None:
    subprocess.run(["bash", "-o", "pipefail", "-c", f"lbzip2 -n {threads} -dc {fastq_bz2} > {destination}"], check=True)


def run_salmon(fastq: Path, out: Path, index: Path, threads: int) -> float:
    """salmon quant on one single-end run. Returns seconds (NaN if quant.sf already exists).

    salmon 2.x needs a seekable input (a pipe fails with "Illegal seek"), so a `.bz2`
    input is decompressed to a temporary file next to `out`. Selective alignment is
    the 2.x default, so the runbook's --validateMappings (a deprecated no-op) is omitted.
    """
    if (out / "quant.sf").exists():
        return float("nan")
    start = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    plain = out.parent / f"{out.name}.tmp.fastq"
    try:
        if fastq.suffix == ".bz2":
            decompress(fastq, plain)
            reads = plain
        else:
            reads = fastq
        subprocess.run(
            ["salmon", "quant", "-i", str(index), "-l", "A", "-r", str(reads), "-p", str(threads), "-o", str(out)],
            check=True,
            capture_output=True,
        )
    finally:
        plain.unlink(missing_ok=True)
    return time.time() - start


def salmon_qc(out: Path) -> dict:
    meta = json.loads((out / "aux_info" / "meta_info.json").read_text())
    lib = json.loads((out / "lib_format_counts.json").read_text())
    return {
        "salmon_version": meta.get("salmon_version"),
        "num_processed": meta.get("num_processed"),
        "num_mapped": meta.get("num_mapped"),
        "percent_mapped": meta.get("percent_mapped"),
        "library_type_detected": lib.get("expected_format"),
        "compatible_fragment_ratio": lib.get("compatible_fragment_ratio"),
        "strand_mapping_bias": lib.get("strand_mapping_bias"),
    }


def parse_bowtie2_log(log: str) -> dict:
    total = int(re.search(r"(\d+) reads; of these", log).group(1))
    unaligned = int(re.search(r"(\d+) \([\d.]+%\) aligned 0 times", log).group(1))
    overall = float(re.search(r"([\d.]+)% overall alignment rate", log).group(1))
    return {"reads_total": total, "reads_aligned": total - unaligned, "overall_alignment_rate": overall / 100}


def align_atac(run: str, fastq: Path, out: Path, index: Path, threads: int) -> dict:
    """bowtie2 -> MAPQ >= 30 -> sort -> markdup -r -> reads BED. Returns alignment QC."""
    out.mkdir(parents=True, exist_ok=True)
    sorted_bam, dedup_bam, bed = out / f"{run}.sorted.bam", out / f"{run}.dedup.bam", out / f"{run}.reads.bed"
    log_path = out / f"{run}.bowtie2.log"
    timing = {}
    if not bed.exists():
        start = time.time()
        sh(
            f"lbzip2 -n 4 -dc {fastq} "
            f"| bowtie2 -p {threads} --very-sensitive -x {index} -U - 2> {log_path} "
            f"| samtools view -b -q 30 -F 4 - "
            f"| samtools sort -@ 4 -o {sorted_bam} - 2> /dev/null"
        )
        timing["align_seconds"] = time.time() - start
        start = time.time()
        sh(f"samtools markdup -r -s -@ 4 {sorted_bam} {dedup_bam} 2> {out / f'{run}.markdup.log'}")
        sh(f"samtools index {dedup_bam}")
        sh(f"samtools view -u {dedup_bam} | bedtools bamtobed -i stdin > {bed}.part && mv {bed}.part {bed}")
        timing["dedup_bed_seconds"] = time.time() - start
    qc = {"run": run, **parse_bowtie2_log(log_path.read_text())}
    mapq30 = int(sh(f"samtools view -c {sorted_bam}").strip())
    dedup = int(sh(f"samtools view -c {dedup_bam}").strip())
    qc.update(
        {
            "reads_mapq30": mapq30,
            "fraction_mapq30_of_aligned": mapq30 / qc["reads_aligned"],
            "reads_dedup": dedup,
            "duplication_rate": 1 - dedup / mapq30,
            **timing,
        }
    )
    return qc


def align_rna_genome(run: str, fastq: Path, out: Path, index: Path, threads: int) -> dict:
    """Unspliced genome alignment of RNA reads for interval counting: bowtie2 -> MAPQ >= 30 -> BED."""
    out.mkdir(parents=True, exist_ok=True)
    bed, log_path = out / f"{run}.rna_genome.bed", out / f"{run}.rna_genome.bowtie2.log"
    start = time.time()
    if not bed.exists():
        sh(
            f"lbzip2 -n 4 -dc {fastq} "
            f"| bowtie2 -p {threads} --very-sensitive -x {index} -U - 2> {log_path} "
            f"| samtools view -u -q 30 -F 4 - "
            f"| bedtools bamtobed -i stdin > {bed}.part && mv {bed}.part {bed}"
        )
    qc = {"run": run, **parse_bowtie2_log(log_path.read_text()), "genome_align_seconds": time.time() - start}
    return qc


def downsample_fastq_records(lines, total: int, n: int, seed: int):
    """Yield the lines of `n` FASTQ records chosen uniformly without replacement from `total` records."""
    if n > total:
        raise ValueError(f"Cannot keep {n} of {total} records")
    keep = np.zeros(total, dtype=bool)
    keep[np.random.default_rng(seed).choice(total, n, replace=False)] = True
    record = 0
    for i, line in enumerate(lines):
        if record >= total:
            raise ValueError("FASTQ has more records than `total`")
        if keep[record]:
            yield line
        if i % 4 == 3:
            record += 1
    if record != total:
        raise ValueError(f"FASTQ has {record} records, expected {total}")


HALF_SEED = b"42:"


def read_half(read_name: str) -> str:
    """Split-half assignment (amendment 01 A): lowest bit of blake2b(b"42:" + name), 0 -> "A", 1 -> "B"."""
    digest = hashlib.blake2b(HALF_SEED + read_name.encode(), digest_size=8).digest()
    return "A" if int.from_bytes(digest, "big") & 1 == 0 else "B"


def split_fastq_halves(lines, half_a, half_b) -> dict[str, int]:
    """Write each 4-line FASTQ record to `half_a` or `half_b` by `read_half` of its name."""
    counts = {"A": 0, "B": 0}
    record = []
    for line in lines:
        record.append(line)
        if len(record) == 4:
            name = record[0][1:].split()[0]
            half = read_half(name)
            (half_a if half == "A" else half_b).writelines(record)
            counts[half] += 1
            record = []
    if record:
        raise ValueError("FASTQ ends with an incomplete record")
    return counts
