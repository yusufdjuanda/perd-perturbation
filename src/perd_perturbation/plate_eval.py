"""Plate-level evaluation under runbook amendment 02 (definitions fixed in execution.md, 2026-10-02)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def binomial_split(counts: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split stored cut-site counts into two halves: half A ~ Binomial(c, 0.5), half B = c - half A."""
    counts = np.asarray(counts, dtype=np.int64)
    half_a = np.random.default_rng(seed).binomial(counts, 0.5)
    return half_a, counts - half_a


def weighted_interval_counts(cuts: pd.DataFrame, intervals: pd.DataFrame) -> np.ndarray:
    """Sum of `count` over cut positions with start <= pos < end, per interval (cuts: chrom, pos, count)."""
    totals = np.zeros(len(intervals), dtype=np.int64)
    by_chrom = {chrom: group for chrom, group in cuts.groupby("chrom", sort=False)}
    for chrom, idx in intervals.groupby("chrom").indices.items():
        if chrom not in by_chrom:
            continue
        group = by_chrom[chrom].sort_values("pos")
        pos = group["pos"].to_numpy()
        cumulative = np.concatenate([[0], np.cumsum(group["count"].to_numpy())])
        lo = np.searchsorted(pos, intervals["start"].to_numpy()[idx], side="left")
        hi = np.searchsorted(pos, intervals["end"].to_numpy()[idx], side="left")
        totals[idx] = cumulative[hi] - cumulative[lo]
    return totals


def loo_center(matrix: np.ndarray) -> np.ndarray:
    """Leave-one-out centering over rows (conditions): x_k - mean over j != k of x_j, per column."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    return matrix - (matrix.sum(axis=0, keepdims=True) - matrix) / (n - 1)


def strata(dmso_openness: np.ndarray, n_genes_used: np.ndarray) -> np.ndarray:
    """Stratum id = DMSO openness decile (rank-based, ties by order) x n_genes_used tercile (rank-based)."""
    def bins(values: np.ndarray, q: int) -> np.ndarray:
        ranks = pd.Series(values).rank(method="first").to_numpy()
        return pd.qcut(ranks, q, labels=False)

    return bins(np.asarray(dmso_openness), 10) * 3 + bins(np.asarray(n_genes_used), 3)


def k_values(n_retained: int, fractions=(0.01, 0.005, 0.02, 0.05), minimum: int = 100) -> dict[float, int]:
    """K per fraction: max(minimum, round(f * n)), capped so Top and Bottom do not overlap."""
    return {f: int(min(max(minimum, round(f * n_retained)), n_retained // 2)) for f in fractions}


def _extreme_from_order(order: np.ndarray, measures: np.ndarray, ks: dict[float, int]) -> np.ndarray:
    """S for every K and measure column, given the ascending order of predictions."""
    cumulative = np.vstack([np.zeros(measures.shape[1]), np.cumsum(measures[order], axis=0)])
    total = cumulative[-1]
    out = np.empty((len(ks), measures.shape[1]))
    for i, k in enumerate(ks.values()):
        top = (total - cumulative[-k - 1]) / k
        bottom = cumulative[k] / k
        out[i] = top - bottom
    return out


def extreme_test(
    predicted: np.ndarray,
    measures: np.ndarray,
    stratum: np.ndarray,
    ks: dict[float, int],
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Amendment 02 C1: S = mean measured over the K highest predictions - over the K lowest.

    predicted: (N,), assumed already in enhancer_id order (ties keep that order).
    measures: (N, M) measured-change columns evaluated with the same permutations.
    Null: predictions shuffled within strata, `n_perm` times. Returns S (K x M), null mean and
    SD (K x M), z, and the one-sided p = (1 + #null >= S) / (n_perm + 1).
    """
    predicted = np.asarray(predicted, dtype=float)
    measures = np.asarray(measures, dtype=float).reshape(len(predicted), -1)
    observed = _extreme_from_order(np.argsort(predicted, kind="stable"), measures, ks)
    rng = np.random.default_rng(seed)
    by_stratum = np.argsort(stratum, kind="stable")
    null = np.empty((n_perm, *observed.shape))
    permuted = np.empty_like(predicted)
    for i in range(n_perm):
        shuffled = np.lexsort((rng.random(len(predicted)), stratum))
        permuted[by_stratum] = predicted[shuffled]
        null[i] = _extreme_from_order(np.argsort(permuted, kind="stable"), measures, ks)
    mean, sd = null.mean(axis=0), null.std(axis=0, ddof=1)
    return {
        "S": observed,
        "null_mean": mean,
        "null_sd": sd,
        "z": (observed - mean) / sd,
        "p_one_sided": (1 + (null >= observed).sum(axis=0)) / (n_perm + 1),
    }


def bootstrap_mean(values: np.ndarray, n_boot: int = 10_000, seed: int = 42, level: float = 0.95) -> dict:
    """Mean with a percentile bootstrap interval over units (conditions), and the bootstrap SE."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    se = float(draws.std(ddof=1))
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "level": level, "bootstrap_se": se, "min_detectable_effect": 2.8 * se, "n": len(values)}


def bootstrap_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42, level: float = 0.95) -> dict:
    """Difference of means a - b with resampling within each group."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    draws = a[rng.integers(0, len(a), (n_boot, len(a)))].mean(axis=1) - b[rng.integers(0, len(b), (n_boot, len(b)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return {"difference": float(a.mean() - b.mean()), "ci_low": float(low), "ci_high": float(high), "level": level, "n_a": len(a), "n_b": len(b)}


def extreme_statistics(predicted: np.ndarray, measured: np.ndarray, k: int) -> tuple[float, float, float]:
    """(S, top mean, bottom mean) for one K, ranking by `predicted` (stable, input order breaks ties)."""
    order = np.argsort(predicted, kind="stable")
    top, bottom = measured[order[-k:]].mean(), measured[order[:k]].mean()
    return float(top - bottom), float(top), float(bottom)


def extreme_test_components(
    predicted: np.ndarray,
    measured: np.ndarray,
    stratum: np.ndarray,
    ks: dict[float, int],
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[float, dict]:
    """Like `extreme_test` for one measured vector, plus opening (top-K mean) and closing
    (negated bottom-K mean) components, each with its own z against the same permutations."""
    predicted, measured = np.asarray(predicted, dtype=float), np.asarray(measured, dtype=float)
    rng = np.random.default_rng(seed)
    by_stratum = np.argsort(stratum, kind="stable")
    observed = {f: extreme_statistics(predicted, measured, k) for f, k in ks.items()}
    null = {f: np.empty((n_perm, 3)) for f in ks}
    permuted = np.empty_like(predicted)
    for i in range(n_perm):
        shuffled = np.lexsort((rng.random(len(predicted)), stratum))
        permuted[by_stratum] = predicted[shuffled]
        order = np.argsort(permuted, kind="stable")
        sorted_measured = measured[order]
        for f, k in ks.items():
            top, bottom = sorted_measured[-k:].mean(), sorted_measured[:k].mean()
            null[f][i] = (top - bottom, top, bottom)
    out = {}
    for f, k in ks.items():
        s, top, bottom = observed[f]
        mean, sd = null[f].mean(axis=0), null[f].std(axis=0, ddof=1)
        out[f] = {
            "K": k,
            "S": s,
            "null_mean": float(mean[0]),
            "null_sd": float(sd[0]),
            "z": float((s - mean[0]) / sd[0]),
            "p_one_sided": float((1 + (null[f][:, 0] >= s).sum()) / (n_perm + 1)),
            "top_mean": top,
            "z_opening": float((top - mean[1]) / sd[1]),
            "bottom_mean": bottom,
            "z_closing": float((mean[2] - bottom) / sd[2]),
        }
    return out
