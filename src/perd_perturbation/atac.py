"""ATAC reads to Tn5 cut-site bins and PERD-style enhancer openness (runbook section 5c).

Coordinates: reads and cut sites are 0-based (BED). Bin k holds cut sites with
position in [200k, 200k + 199] and is represented as ../perd represents its bins,
start = 200k and end = start + 199 (closed). PERD's containment rule is applied to
these bins and the unshifted GeneHancer coordinates exactly as ../perd does
(execution.md, 2026-10-01 coordinate revision).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation.vendor.perd_repro.openness import compute_enhancer_openness

BIN_SIZE = 200
BIRD_N = 17_002_867  # BIRD cohort minimum total read count (Zhou et al. 2017)
CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
TN5_PLUS_SHIFT = 4
TN5_MINUS_SHIFT = -5


def read_bed_reads(path: Path | str) -> pd.DataFrame:
    """Read `bedtools bamtobed` output (chrom, start, end, name, score, strand)."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 5],
        names=["chrom", "start", "end", "strand"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64, "strand": str},
    )


def merge_intervals(intervals: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Merge overlapping or touching intervals per chromosome -> {chrom: (starts, ends)} sorted."""
    merged = {}
    for chrom, group in intervals.groupby("chrom", sort=False):
        order = np.argsort(group["start"].to_numpy(), kind="stable")
        starts = group["start"].to_numpy()[order]
        ends = group["end"].to_numpy()[order]
        out_s, out_e = [starts[0]], [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s <= out_e[-1]:
                out_e[-1] = max(out_e[-1], e)
            else:
                out_s.append(s)
                out_e.append(e)
        merged[chrom] = (np.asarray(out_s, dtype=np.int64), np.asarray(out_e, dtype=np.int64))
    return merged


def overlaps_merged(starts: np.ndarray, ends: np.ndarray, merged_starts: np.ndarray, merged_ends: np.ndarray) -> np.ndarray:
    """True where half-open [start, end) overlaps any merged, sorted, disjoint interval."""
    idx = np.searchsorted(merged_starts, ends, side="left") - 1
    valid = idx >= 0
    result = np.zeros(len(starts), dtype=bool)
    result[valid] = merged_ends[idx[valid]] > starts[valid]
    return result


def filter_reads(reads: pd.DataFrame, blacklist: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep reads on chr1-22/chrX that do not overlap the blacklist. Returns (reads, counts)."""
    counts = {"reads_in": len(reads)}
    canonical = reads[reads["chrom"].isin(CANONICAL)]
    counts["reads_chrM"] = int((reads["chrom"] == "chrM").sum())
    counts["reads_noncanonical_removed"] = len(reads) - len(canonical)
    merged = merge_intervals(blacklist)
    keep = np.ones(len(canonical), dtype=bool)
    chroms = canonical["chrom"].to_numpy()
    starts = canonical["start"].to_numpy()
    ends = canonical["end"].to_numpy()
    for chrom, (bl_s, bl_e) in merged.items():
        mask = chroms == chrom
        if mask.any():
            keep[mask] = ~overlaps_merged(starts[mask], ends[mask], bl_s, bl_e)
    kept = canonical[keep].reset_index(drop=True)
    counts["reads_blacklist_removed"] = int((~keep).sum())
    counts["reads_final"] = len(kept)
    return kept, counts


def tn5_cut_sites(reads: pd.DataFrame) -> pd.DataFrame:
    """Tn5 cut site per read: 5' end shifted +4 on the plus strand and -5 on the minus strand.

    Plus-strand 5' end is `start`; minus-strand 5' end is `end - 1` (0-based).
    """
    plus = reads["strand"].to_numpy() == "+"
    pos = five_prime_ends(reads)["pos"].to_numpy() + np.where(plus, TN5_PLUS_SHIFT, TN5_MINUS_SHIFT)
    return pd.DataFrame({"chrom": reads["chrom"].to_numpy(), "pos": np.maximum(pos, 0)})


def five_prime_ends(reads: pd.DataFrame) -> pd.DataFrame:
    """0-based 5' end per read: `start` on the plus strand, `end - 1` on the minus strand."""
    plus = reads["strand"].to_numpy() == "+"
    return pd.DataFrame(
        {"chrom": reads["chrom"].to_numpy(), "pos": np.where(plus, reads["start"].to_numpy(), reads["end"].to_numpy() - 1)}
    )


def bin_counts(cuts: pd.DataFrame, bin_size: int = BIN_SIZE) -> pd.DataFrame:
    """Sparse cut-site counts per bin: columns chrom, bin (floor(pos / bin_size)), count."""
    binned = cuts.assign(bin=cuts["pos"].to_numpy() // bin_size)
    counts = binned.groupby(["chrom", "bin"], sort=True).size().rename("count").reset_index()
    counts["count"] = counts["count"].astype(np.int64)
    return counts


def save_bins(path: Path, counts: pd.DataFrame, meta: dict[str, int]) -> None:
    np.savez_compressed(
        path,
        chrom=counts["chrom"].to_numpy().astype("U5"),
        bin=counts["bin"].to_numpy(),
        count=counts["count"].to_numpy(),
        meta_keys=np.array(list(meta), dtype="U64"),
        meta_values=np.array(list(meta.values()), dtype=np.int64),
    )


def load_bins(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    data = np.load(path)
    counts = pd.DataFrame({"chrom": data["chrom"].astype(str), "bin": data["bin"], "count": data["count"]})
    meta = dict(zip(data["meta_keys"].astype(str), data["meta_values"].tolist()))
    return counts, meta


def normalize(counts: np.ndarray | pd.Series, total_reads: int, target: int = BIRD_N):
    """BIRD depth normalization then log: log2(N * n / N_i + 1)."""
    return np.log2(target * np.asarray(counts, dtype=float) / total_reads + 1)


def bin_interval(bins: np.ndarray, bin_size: int = BIN_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """../perd bin representation: start = k * bin_size, end = start + bin_size - 1."""
    return bins * bin_size, bins * bin_size + bin_size - 1


def contained_grid_bins(enhancers: pd.DataFrame, bin_size: int = BIN_SIZE) -> pd.DataFrame:
    """All grid bins with bin.start >= enh.start and bin.end <= enh.end: columns chrom, bin (unique, sorted)."""
    first = -(-enhancers["start"].to_numpy() // bin_size)  # smallest k with k * size >= start
    last = (enhancers["end"].to_numpy() - (bin_size - 1)) // bin_size  # largest k with k * size + size - 1 <= end
    n = np.maximum(last - first + 1, 0)
    chroms = np.repeat(enhancers["chrom"].to_numpy(), n)
    offsets = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    bins = np.repeat(first, n) + offsets
    return pd.DataFrame({"chrom": chroms, "bin": bins}).drop_duplicates().sort_values(["chrom", "bin"]).reset_index(drop=True)


def enhancer_openness(
    enhancers: pd.DataFrame,
    normalized_bins: dict[str, pd.DataFrame],
    bin_size: int = BIN_SIZE,
) -> pd.DataFrame:
    """PERD enhancer openness for each run: max normalized bin value over strictly contained bins.

    enhancers: GeneHancer table (enhancer_id, chrom, start, end) as stored in ../perd, unshifted.
    normalized_bins: run -> sparse frame (chrom, bin, value) of log2-normalized signal.

    Every contained grid bin is materialized (zeros included) and passed to the
    vendored `compute_enhancer_openness`, so enhancers whose bins have no cut sites
    get 0 rather than dropping out. Enhancers off chr1-22/chrX (reads there are
    discarded) and enhancers containing no complete bin are NaN.
    """
    measurable = enhancers[enhancers["chrom"].isin(CANONICAL)]
    grid = contained_grid_bins(measurable, bin_size)
    dense = grid.copy()
    for run, values in normalized_bins.items():
        merged = grid.merge(values[["chrom", "bin", "value"]], on=["chrom", "bin"], how="left")
        dense[run] = merged["value"].fillna(0.0).to_numpy()
    starts, ends = bin_interval(dense["bin"].to_numpy(), bin_size)
    dense = dense.assign(start=starts, end=ends).rename(columns={"chrom": "chromosome"})
    dense = dense[["chromosome", "start", "end", *normalized_bins]]
    openness, _ = compute_enhancer_openness(measurable[["enhancer_id", "chrom", "start", "end"]], dense)
    return openness.reindex(enhancers["enhancer_id"])


# ---- QC helpers -----------------------------------------------------------


def tss_table(gtf_gz: Path | str) -> pd.DataFrame:
    """Gene-level TSSs (0-based) from a GENCODE GTF: columns chrom, pos, strand, gene_id, gene_name, gene_type."""
    genes = pd.read_csv(
        gtf_gz,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 2, 3, 4, 6, 8],
        names=["chrom", "feature", "start", "end", "strand", "attributes"],
    )
    genes = genes[genes["feature"] == "gene"]
    attrs = genes["attributes"]
    tss = pd.DataFrame(
        {
            "chrom": genes["chrom"].to_numpy(),
            "pos": np.where(genes["strand"] == "+", genes["start"] - 1, genes["end"] - 1),
            "strand": genes["strand"].to_numpy(),
            "gene_id": attrs.str.extract(r'gene_id "([^"]+)"')[0].str.split(".").str[0].to_numpy(),
            "gene_name": attrs.str.extract(r'gene_name "([^"]+)"')[0].to_numpy(),
            "gene_type": attrs.str.extract(r'gene_type "([^"]+)"')[0].to_numpy(),
        }
    )
    return tss.reset_index(drop=True)


def fraction_in_intervals(cuts: pd.DataFrame, intervals: pd.DataFrame) -> float:
    """Fraction of cut sites inside any interval (0-based half-open)."""
    merged = merge_intervals(intervals)
    inside = 0
    for chrom, group in cuts.groupby("chrom", sort=False):
        if chrom in merged:
            pos = group["pos"].to_numpy()
            inside += int(overlaps_merged(pos, pos + 1, *merged[chrom]).sum())
    return inside / len(cuts)


def tss_profile(cuts: pd.DataFrame, tss: pd.DataFrame, flank: int = 2000) -> np.ndarray:
    """Strand-aware aggregate cut-site counts at offsets -flank..+flank around TSSs (length 2*flank+1)."""
    profile = np.zeros(2 * flank + 1, dtype=np.int64)
    tss = tss.drop_duplicates(["chrom", "pos", "strand"])
    for chrom, sites in tss.groupby("chrom", sort=False):
        pos = np.sort(cuts.loc[cuts["chrom"] == chrom, "pos"].to_numpy())
        if len(pos) == 0:
            continue
        centers = sites["pos"].to_numpy()
        signs = np.where(sites["strand"].to_numpy() == "+", 1, -1)
        lo = np.searchsorted(pos, centers - flank, side="left")
        hi = np.searchsorted(pos, centers + flank, side="right")
        n = hi - lo
        if n.sum() == 0:
            continue
        idx = np.repeat(lo, n) + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
        offsets = (pos[idx] - np.repeat(centers, n)) * np.repeat(signs, n)
        profile += np.bincount(offsets + flank, minlength=2 * flank + 1)
    return profile


def tss_enrichment(profile: np.ndarray, end_width: int = 100, smooth: int = 51) -> float:
    """ENCODE-style TSS enrichment: profile / mean of the outer `end_width` bp on both sides,
    smoothed with a centered rolling mean, maximum within +-1 kb of the TSS."""
    background = np.concatenate([profile[:end_width], profile[-end_width:]]).mean()
    if background == 0:
        return float("nan")
    smoothed = pd.Series(profile / background).rolling(smooth, center=True, min_periods=1).mean().to_numpy()
    flank = (len(profile) - 1) // 2
    return float(smoothed[flank - 1000 : flank + 1001].max())


def promoter_counts(cuts: pd.DataFrame, tss: pd.DataFrame, window: int = 1000) -> pd.Series:
    """Cut sites within +-window of each TSS, indexed by gene_name (first TSS per name)."""
    tss = tss.drop_duplicates("gene_name")
    result = np.zeros(len(tss), dtype=np.int64)
    positions = {chrom: np.sort(group["pos"].to_numpy()) for chrom, group in cuts.groupby("chrom", sort=False)}
    chroms = tss["chrom"].to_numpy()
    centers = tss["pos"].to_numpy()
    for chrom in np.unique(chroms):
        if chrom not in positions:
            continue
        mask = chroms == chrom
        pos = positions[chrom]
        result[mask] = np.searchsorted(pos, centers[mask] + window, side="right") - np.searchsorted(
            pos, centers[mask] - window, side="left"
        )
    return pd.Series(result, index=tss["gene_name"].to_numpy(), name="promoter_cuts")


# ---- base-pair cut-site storage ----------------------------------------------


def save_cut_sites(path: Path, cuts: pd.DataFrame, meta: dict[str, int]) -> None:
    """Store cut sites as unique sorted (chrom, pos) with counts, chromosome blocks in CANONICAL order."""
    counted = cuts.groupby(["chrom", "pos"], sort=False).size().rename("count").reset_index()
    order = {chrom: i for i, chrom in enumerate(CANONICAL)}
    counted = counted.assign(rank=counted["chrom"].map(order)).sort_values(["rank", "pos"])
    chroms = [c for c in CANONICAL if c in set(counted["chrom"])]
    sizes = counted.groupby("chrom", sort=False).size().reindex(chroms).to_numpy()
    if counted["count"].max() > np.iinfo(np.uint16).max:
        raise ValueError("cut-site count exceeds uint16")
    np.savez_compressed(
        path,
        chrom=np.array(chroms, dtype="U5"),
        offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        pos=counted["pos"].to_numpy().astype(np.int32),
        count=counted["count"].to_numpy().astype(np.uint16),
        meta_keys=np.array(list(meta), dtype="U64"),
        meta_values=np.array(list(meta.values()), dtype=np.int64),
    )


def load_cut_sites(path: Path, expand: bool = False) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load cut sites: columns chrom, pos, count (or one row per cut with expand=True)."""
    data = np.load(path)
    offsets = data["offsets"]
    chrom = np.repeat(data["chrom"].astype(str), np.diff(offsets))
    frame = pd.DataFrame({"chrom": chrom, "pos": data["pos"].astype(np.int64), "count": data["count"].astype(np.int64)})
    if expand:
        frame = frame.loc[frame.index.repeat(frame["count"])].drop(columns="count").reset_index(drop=True)
    meta = dict(zip(data["meta_keys"].astype(str), data["meta_values"].tolist()))
    return frame, meta
