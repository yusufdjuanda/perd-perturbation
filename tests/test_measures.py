import numpy as np
import pytest

from perd_perturbation import measures
from perd_perturbation.runs import read_half, split_fastq_halves


def test_tmm_ignores_global_scaling_by_depth():
    rng = np.random.default_rng(0)
    reference = rng.poisson(50, size=2000) + 1
    assert measures.tmm_factor(reference * 3, reference) == pytest.approx(1.0)
    assert measures.library_ratio(reference * 3, reference) == pytest.approx(3.0)


def test_tmm_is_robust_to_a_subset_opening():
    rng = np.random.default_rng(1)
    expected = rng.uniform(20, 400, size=5000)
    reference = rng.poisson(expected) + 1
    opened = expected.copy()
    opened[:500] *= 20  # 10% of features open strongly
    test = rng.poisson(opened) + 1
    # Raw library ratio is inflated by the opening; TMM restores the unchanged majority to M = 0.
    s = measures.library_ratio(test, reference)
    raw = test.sum() / reference.sum()
    assert raw > 2.5
    assert s == pytest.approx(1.0, rel=0.05)


def test_delta_count_formula():
    assert measures.delta_count([7], [3], 2.0)[0] == pytest.approx(np.log2(8 / 7))


def test_read_half_is_deterministic_and_balanced(tmp_path):
    names = [f"DRR1.{i}" for i in range(4000)]
    halves = [read_half(n) for n in names]
    assert halves == [read_half(n) for n in names]
    assert 0.45 < halves.count("A") / len(halves) < 0.55
    lines = [line for n in names[:10] for line in (f"@{n} len=4\n", "ACGT\n", "+\n", "IIII\n")]
    with open(tmp_path / "a.fq", "w") as a, open(tmp_path / "b.fq", "w") as b:
        counts = split_fastq_halves(lines, a, b)
    assert counts["A"] + counts["B"] == 10
    assert (tmp_path / "a.fq").read_text().count("@") == counts["A"]


def test_spearman_bootstrap_and_spearman_brown():
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    result = measures.spearman_bootstrap(x, x + rng.normal(scale=0.5, size=500), n_boot=200)
    assert 0.8 < result["spearman"] < 0.95
    assert result["ci_low"] < result["spearman"] < result["ci_high"]
    assert measures.spearman_brown(0.5) == pytest.approx(2 / 3)
