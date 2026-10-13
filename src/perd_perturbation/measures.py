"""Measured-change definitions for section 8 (runbook amendment 01, C1 and C2)."""

from __future__ import annotations

import numpy as np


def tmm_factor(test: np.ndarray, reference: np.ndarray, logratio_trim: float = 0.3, sum_trim: float = 0.05) -> float:
    """edgeR TMM normalization factor of `test` relative to `reference` (calcNormFactors, method="TMM").

    Library sizes are the column sums. Features with a zero count in either sample are
    dropped. `logratio_trim` and `sum_trim` are trimmed from each end, as edgeR does, and the
    remaining log ratios are averaged with edgeR's precision weights. Returns 2^(weighted mean M).
    """
    test, reference = np.asarray(test, dtype=float), np.asarray(reference, dtype=float)
    n_test, n_ref = test.sum(), reference.sum()
    keep = (test > 0) & (reference > 0)
    t, r = test[keep], reference[keep]
    m = np.log2((t / n_test) / (r / n_ref))
    a = 0.5 * np.log2((t / n_test) * (r / n_ref))
    variance = (n_test - t) / n_test / t + (n_ref - r) / n_ref / r
    n = len(m)
    lo_m, hi_m = int(np.floor(n * logratio_trim)) + 1, n + 1 - (int(np.floor(n * logratio_trim)) + 1)
    lo_a, hi_a = int(np.floor(n * sum_trim)) + 1, n + 1 - (int(np.floor(n * sum_trim)) + 1)
    rank_m = _rank(m)
    rank_a = _rank(a)
    selected = (rank_m >= lo_m) & (rank_m <= hi_m) & (rank_a >= lo_a) & (rank_a <= hi_a)
    if not selected.any():
        raise ValueError("TMM trimming left no features (heavily tied log ratios)")
    weighted = np.sum(m[selected] / variance[selected]) / np.sum(1 / variance[selected])
    return float(2**weighted)


def _rank(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (R's rank default, as used by edgeR)."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            ranks[order[start:end]] = (start + 1 + end) / 2
            start = end
    return ranks


def library_ratio(test: np.ndarray, reference: np.ndarray) -> float:
    """s = (N_test * f_test) / N_reference, scaling reference counts to the test library (C2)."""
    test, reference = np.asarray(test, dtype=float), np.asarray(reference, dtype=float)
    return float(test.sum() * tmm_factor(test, reference) / reference.sum())


def delta_count(treated: np.ndarray, control: np.ndarray, s: float) -> np.ndarray:
    """C2 change: log2((c_treated + 1) / (c_control * s + 1))."""
    return np.log2((np.asarray(treated, dtype=float) + 1) / (np.asarray(control, dtype=float) * s + 1))


def spearman_bootstrap(x, y, n_boot: int = 2000, seed: int = 42) -> dict:
    """Spearman correlation with a percentile bootstrap 95% interval over observations."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    estimate = _pearson(rankdata(x), rankdata(y))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        draws[i] = _pearson(rankdata(x[idx]), rankdata(y[idx]))
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {"spearman": float(estimate), "ci_low": float(low), "ci_high": float(high), "n": int(len(x)), "n_boot": n_boot, "seed": seed}


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else np.nan


def spearman_brown(r: float) -> float:
    """Full-depth reliability from a split-half correlation: 2r / (1 + r)."""
    return 2 * r / (1 + r)
