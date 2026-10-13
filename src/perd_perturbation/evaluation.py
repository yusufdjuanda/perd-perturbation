"""Section 8 metrics (runbook amendment 01, C3; definitions fixed in execution.md 2026-10-02)."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator else np.nan


def correlation(x, y, method: str = "spearman") -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if method == "spearman":
        return _pearson(rankdata(x), rankdata(y))
    if method == "pearson":
        return _pearson(x, y)
    raise ValueError(method)


def correlation_bootstrap(x, y, method: str = "spearman", n_boot: int = 2000, seed: int = 42) -> dict:
    """Point estimate and percentile bootstrap 95% interval over observations."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    estimate = correlation(x, y, method)
    if len(x) < 3 or np.isnan(estimate):
        return {"estimate": estimate, "ci_low": np.nan, "ci_high": np.nan, "n": int(len(x))}
    rng = np.random.default_rng(seed)
    draws = np.array([correlation(x[idx], y[idx], method) for idx in (rng.integers(0, len(x), len(x)) for _ in range(n_boot))])
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high), "n": int(len(x))}


def direction_accuracy(predicted, measured, threshold: float, center: bool = False) -> dict:
    """Sign agreement among events |measured| > threshold (an exactly zero prediction counts as wrong).

    center=True subtracts the median of `measured` before events are selected.
    """
    predicted, measured = np.asarray(predicted, dtype=float), np.asarray(measured, dtype=float)
    if center:
        measured = measured - np.median(measured)
    events = np.abs(measured) > threshold
    agree = np.sign(predicted[events]) == np.sign(measured[events])
    opening = measured[events] > 0
    return {
        "n_events": int(events.sum()),
        "n_opening": int(opening.sum()),
        "n_closing": int((~opening).sum()),
        "accuracy": float(agree.mean()) if events.any() else np.nan,
        "accuracy_opening": float(agree[opening].mean()) if opening.any() else np.nan,
        "accuracy_closing": float(agree[~opening].mean()) if (~opening).any() else np.nan,
    }


def event_auc(score, measured, threshold: float, signed: bool = True) -> dict:
    """AUROC/AUPRC for opening (measured > threshold) and closing (measured < -threshold) events.

    Signed predictors score closing with -score; rank-type scores use the raw score for both.
    """
    score, measured = np.asarray(score, dtype=float), np.asarray(measured, dtype=float)
    result = {}
    for direction, labels, s in (
        ("opening", measured > threshold, score),
        ("closing", measured < -threshold, -score if signed else score),
    ):
        prevalence = float(labels.mean())
        if labels.all() or not labels.any():
            result[direction] = {"n_events": int(labels.sum()), "prevalence": prevalence, "auroc": np.nan, "auprc": np.nan}
            continue
        result[direction] = {
            "n_events": int(labels.sum()),
            "prevalence": prevalence,
            "auroc": float(roc_auc_score(labels, s)),
            "auprc": float(average_precision_score(labels, s)),
        }
    return result


def cmh_odds_ratio(exposed: np.ndarray, outcome: np.ndarray, stratum: np.ndarray) -> dict:
    """Cochran-Mantel-Haenszel odds ratio of `outcome` for `exposed` vs not, across strata.

    Returns the Mantel-Haenszel OR with a 95% interval (Robins-Breslow-Greenland variance of log OR),
    and a one-sided p for OR > 1 from the CMH statistic (normal approximation, no continuity correction).
    Strata with no exposed or no unexposed units contribute nothing.
    """
    from scipy.stats import norm

    exposed, outcome = np.asarray(exposed, dtype=bool), np.asarray(outcome, dtype=bool)
    numerator = denominator = 0.0
    p_r = p_s_q_r = q_s = 0.0
    observed = expected = variance = 0.0
    for s in np.unique(stratum):
        m = stratum == s
        a = np.sum(exposed[m] & outcome[m])
        b = np.sum(exposed[m] & ~outcome[m])
        c = np.sum(~exposed[m] & outcome[m])
        d = np.sum(~exposed[m] & ~outcome[m])
        n = a + b + c + d
        if (a + b) == 0 or (c + d) == 0 or n < 2:
            continue
        r, sv = a * d / n, b * c / n
        numerator += r
        denominator += sv
        p, q = (a + d) / n, (b + c) / n
        p_r += p * r
        p_s_q_r += p * sv + q * r
        q_s += q * sv
        observed += a
        expected += (a + b) * (a + c) / n
        variance += (a + b) * (c + d) * (a + c) * (b + d) / (n**2 * (n - 1))
    if denominator == 0 or numerator == 0:
        return {"odds_ratio": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_one_sided": np.nan, "exposed": int(exposed.sum()), "exposed_with_outcome": int((exposed & outcome).sum())}
    odds_ratio = numerator / denominator
    var_log = p_r / (2 * numerator**2) + p_s_q_r / (2 * numerator * denominator) + q_s / (2 * denominator**2)
    se = np.sqrt(var_log)
    z = (observed - expected) / np.sqrt(variance) if variance > 0 else np.nan
    return {
        "odds_ratio": float(odds_ratio),
        "ci_low": float(np.exp(np.log(odds_ratio) - 1.96 * se)),
        "ci_high": float(np.exp(np.log(odds_ratio) + 1.96 * se)),
        "p_one_sided": float(norm.sf(z)) if not np.isnan(z) else np.nan,
        "exposed": int(exposed.sum()),
        "exposed_with_outcome": int((exposed & outcome).sum()),
        "observed": float(observed),
        "expected": float(expected),
    }


def holm(p_values) -> np.ndarray:
    """Holm-adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p_values, dtype=float)
    valid = ~np.isnan(p)
    adjusted = np.full_like(p, np.nan)
    order = np.argsort(p[valid])
    values = p[valid][order]
    m = len(values)
    running = 0.0
    out = np.empty(m)
    for i, v in enumerate(values):
        running = max(running, min(1.0, (m - i) * v))
        out[i] = running
    temp = np.empty(m)
    temp[order] = out
    adjusted[valid] = temp
    return adjusted
