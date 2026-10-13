"""ENCODE A549 dexamethasone time course (runbook amendment 03; details fixed in execution.md 2026-10-04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation import rna

BIN_SIZE = 200


def read_encode_gene_fpkm(path: Path | str) -> pd.Series:
    """FPKM per unversioned Ensembl gene ID from an ENCODE RSEM gene-quantification tsv.

    Non-Ensembl rows (spike-ins) are dropped; PAR_Y copies merge with their X copy (summed).
    """
    table = pd.read_csv(path, sep="\t", usecols=["gene_id", "FPKM"])
    table = table[table["gene_id"].str.startswith("ENSG")]
    fpkm = table.assign(gene_id=table["gene_id"].map(rna.strip_version)).groupby("gene_id")["FPKM"].sum()
    fpkm.index.name = "gene_id"
    return fpkm


def encode_log2fpkm_by_symbol(path: Path | str, symbols: pd.Series) -> tuple[pd.Series, int]:
    """log2(FPKM + 1) by GENCODE symbol, using the DBKERO duplicate rule (lowest Ensembl ID kept).

    Genes absent from `symbols` are dropped. Returns (values, number of dropped duplicate genes).
    """
    fpkm = read_encode_gene_fpkm(path)
    by_symbol, dropped = rna.collapse_to_symbols(fpkm[fpkm.index.isin(symbols.index)], symbols)
    return rna.log2_fpkm(by_symbol), dropped


def chromosome_means(values: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Mean of `values` (one chromosome, NaN = 0) over half-open windows [start, end)."""
    filled = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    cumulative = np.concatenate([[0.0], np.cumsum(filled)])
    starts = np.clip(starts, 0, len(filled))
    ends = np.clip(ends, 0, len(filled))
    return (cumulative[ends] - cumulative[starts]) / (np.asarray(ends) - np.asarray(starts)).clip(min=1)


def bigwig_window_means(bw, windows: pd.DataFrame) -> np.ndarray:
    """Mean signal over each half-open window (chrom, start, end); bases without data count as 0."""
    out = np.zeros(len(windows), dtype=np.float64)
    lengths = bw.chroms()
    for chrom, idx in windows.groupby("chrom").indices.items():
        if chrom not in lengths:
            continue
        values = np.asarray(bw.values(chrom, 0, lengths[chrom], numpy=True))
        sub = windows.iloc[idx]
        means = chromosome_means(values, sub["start"].to_numpy(), sub["end"].to_numpy())
        # Windows running past the chromosome end are averaged over their full nominal length.
        nominal = (sub["end"] - sub["start"]).to_numpy()
        covered = (np.minimum(sub["end"].to_numpy(), lengths[chrom]) - np.maximum(sub["start"].to_numpy(), 0)).clip(min=0)
        out[idx] = means * covered / nominal
    return out


def interval_mean(bw, chrom: str, start: int, end: int) -> float:
    """Verification method: length-weighted mean over [start, end) from bigWig intervals (gaps = 0)."""
    total = 0.0
    for s, e, v in bw.intervals(chrom, start, end) or []:
        total += (min(e, end) - max(s, start)) * v
    return total / (end - start)


def grid_windows(grid: pd.DataFrame) -> pd.DataFrame:
    """Half-open bigWig windows [200k, 200k + 200) for grid bins (chrom, bin)."""
    return pd.DataFrame({"chrom": grid["chrom"].to_numpy(), "start": grid["bin"].to_numpy() * BIN_SIZE, "end": grid["bin"].to_numpy() * BIN_SIZE + BIN_SIZE})


def replicate_halves(replicate_numbers) -> tuple[list[int], list[int]]:
    """Odd vs even positions after sorting by ENCODE biological replicate number (1-based)."""
    ordered = sorted(replicate_numbers)
    return ordered[0::2], ordered[1::2]


def scale_constant(perd_median_positive: float, control_raw_positive_median: float) -> float:
    """k such that log2(k * s + 1) maps the control's positive median onto PERD's positive median."""
    return (2**perd_median_positive - 1) / control_raw_positive_median
