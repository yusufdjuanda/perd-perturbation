import numpy as np
import pytest

from perd_perturbation import evaluation


def test_direction_accuracy_counts_zero_prediction_as_wrong_and_centers_before_events():
    measured = np.array([3.0, -3.0, 3.0, 0.1])
    predicted = np.array([1.0, -1.0, 0.0, 5.0])
    result = evaluation.direction_accuracy(predicted, measured, threshold=2.0)
    assert result["n_events"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    # A global shift: centering moves the median to zero before selecting events.
    shifted = evaluation.direction_accuracy(np.array([1.0, 1.0, -1.0, -1.0]), np.array([5.0, 5.0, 1.0, 1.0]), threshold=1.5, center=True)
    assert shifted["n_events"] == 4
    assert shifted["accuracy"] == 1.0


def test_event_auc_signed_and_rank_type():
    measured = np.array([3.0, 2.5, 0.0, 0.1, -3.0, -2.6])
    score = np.array([2.0, 1.5, 0.0, 0.2, -2.0, -1.0])
    signed = evaluation.event_auc(score, measured, threshold=2.0)
    assert signed["opening"]["auroc"] == 1.0
    assert signed["closing"]["auroc"] == 1.0
    assert signed["opening"]["prevalence"] == pytest.approx(2 / 6)
    rank_type = evaluation.event_auc(score, measured, threshold=2.0, signed=False)
    assert rank_type["closing"]["auroc"] == 0.0  # raw score in both directions: inverse for closing


def test_correlation_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    result = evaluation.correlation_bootstrap(x, 2 * x + rng.normal(size=300), method="pearson", n_boot=200)
    assert result["ci_low"] < result["estimate"] < result["ci_high"]
    assert np.isnan(evaluation.correlation(np.zeros(5), np.arange(5)))


def test_cmh_odds_ratio_matches_single_stratum_and_detects_enrichment():
    # One stratum: MH OR equals the plain OR ad/bc.
    exposed = np.array([1] * 10 + [0] * 90, dtype=bool)
    outcome = np.array([1] * 5 + [0] * 5 + [1] * 9 + [0] * 81, dtype=bool)
    result = evaluation.cmh_odds_ratio(exposed, outcome, np.zeros(100, dtype=int))
    assert result["odds_ratio"] == pytest.approx((5 * 81) / (5 * 9))
    assert result["ci_low"] < result["odds_ratio"] < result["ci_high"]
    assert result["p_one_sided"] < 0.01
    # Confounding by stratum: outcome rate depends only on stratum, exposure too -> OR about 1 after stratification.
    rng = np.random.default_rng(0)
    stratum = rng.integers(0, 2, 20000)
    exposed = rng.random(20000) < np.where(stratum == 1, 0.5, 0.05)
    outcome = rng.random(20000) < np.where(stratum == 1, 0.3, 0.02)
    stratified = evaluation.cmh_odds_ratio(exposed, outcome, stratum)
    crude = evaluation.cmh_odds_ratio(exposed, outcome, np.zeros(20000, dtype=int))
    assert crude["odds_ratio"] > 2
    assert 0.8 < stratified["odds_ratio"] < 1.25


def test_holm():
    adjusted = evaluation.holm([0.01, 0.04, 0.03, np.nan])
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    assert np.isnan(adjusted[3])
