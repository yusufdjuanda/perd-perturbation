import numpy as np
import pandas as pd
import pytest

from perd_perturbation import plate_eval


def test_binomial_split_preserves_counts_and_is_seeded():
    counts = np.array([0, 1, 5, 10, 100])
    a, b = plate_eval.binomial_split(counts)
    assert (a + b == counts).all()
    assert (a == plate_eval.binomial_split(counts)[0]).all()


def test_weighted_interval_counts_half_open():
    cuts = pd.DataFrame({"chrom": ["chr1", "chr1", "chr1", "chr2"], "pos": [10, 19, 20, 5], "count": [2, 1, 4, 3]})
    intervals = pd.DataFrame({"chrom": ["chr1", "chr2", "chr3"], "start": [10, 0, 0], "end": [20, 6, 1]})
    assert plate_eval.weighted_interval_counts(cuts, intervals).tolist() == [3, 3, 0]


def test_loo_center_removes_shared_component():
    shared = np.array([1.0, -2.0, 3.0])
    matrix = np.vstack([shared + 0.0, shared + 1.0, shared - 1.0])
    centered = plate_eval.loo_center(matrix)
    # Row 1: (s + 1) - mean(s + 0, s - 1) = 1.5, whatever the shared component.
    np.testing.assert_allclose(centered[1], 1.5)


def test_extreme_test_detects_signal_and_null_is_calibrated():
    rng = np.random.default_rng(0)
    n = 5000
    stratum = plate_eval.strata(rng.normal(size=n), rng.integers(2, 50, size=n))
    signal = rng.normal(size=n)
    measured = np.column_stack([signal + rng.normal(scale=2, size=n), rng.normal(size=n)])
    ks = plate_eval.k_values(n)
    result = plate_eval.extreme_test(signal, measured, stratum, ks, n_perm=300)
    assert result["S"].shape == (4, 2)
    assert result["z"][0, 0] > 4  # informative column
    assert abs(result["z"][0, 1]) < 3.5  # pure-noise column


def test_stratified_null_removes_a_baseline_artifact():
    # Measured change depends only on the stratum variable (regression to the mean);
    # a predictor equal to that variable must not score once permuted within strata.
    rng = np.random.default_rng(1)
    n = 5000
    baseline = rng.normal(size=n)
    measured = (-baseline + rng.normal(scale=0.5, size=n)).reshape(-1, 1)
    stratum = plate_eval.strata(baseline, rng.integers(2, 50, size=n))
    naive = plate_eval.extreme_test(-baseline, measured, np.zeros(n, dtype=int), plate_eval.k_values(n), n_perm=300)
    stratified = plate_eval.extreme_test(-baseline, measured, stratum, plate_eval.k_values(n), n_perm=300)
    assert naive["z"][0, 0] > 10
    # Decile strata remove only part of a strong, continuous baseline dependence (residual
    # within-decile slope); amendment 02 reports the DMSO-openness diagnostic for this reason.
    assert stratified["z"][0, 0] < 0.75 * naive["z"][0, 0]


def test_k_values_minimum_and_cap():
    assert plate_eval.k_values(50_000)[0.01] == 500
    assert plate_eval.k_values(3_000)[0.01] == 100
    assert plate_eval.k_values(150)[0.05] == 75


def test_bootstrap_mean_and_difference():
    values = np.array([0.5, 1.0, 1.5, 2.0])
    result = plate_eval.bootstrap_mean(values, n_boot=2000, level=0.975)
    assert result["mean"] == pytest.approx(1.25)
    assert result["ci_low"] < 1.25 < result["ci_high"]
    assert result["min_detectable_effect"] == pytest.approx(2.8 * result["bootstrap_se"])
    diff = plate_eval.bootstrap_difference(values, values - 1, n_boot=2000)
    assert diff["difference"] == pytest.approx(1.0)


def test_extreme_test_components_match_extreme_test_and_split_by_direction():
    rng = np.random.default_rng(2)
    n = 4000
    stratum = plate_eval.strata(rng.normal(size=n), rng.integers(2, 50, size=n))
    predicted = rng.normal(size=n)
    # Only the opening side is informative: the top 1% of predictions truly open; nothing else moves.
    measured = 3.0 * (predicted > np.quantile(predicted, 0.99)) + rng.normal(scale=1.0, size=n)
    ks = plate_eval.k_values(n)
    components = plate_eval.extreme_test_components(predicted, measured, stratum, ks, n_perm=300)
    reference = plate_eval.extreme_test(predicted, measured.reshape(-1, 1), stratum, ks, n_perm=300)
    first = components[0.01]
    assert first["S"] == pytest.approx(reference["S"][0, 0])
    assert first["z"] == pytest.approx(reference["z"][0, 0])
    assert first["z_opening"] > 5
    assert abs(first["z_closing"]) < 3.5
