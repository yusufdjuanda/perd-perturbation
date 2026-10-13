import numpy as np
import pandas as pd

from perd_perturbation.vendor.perd_repro.models import fit_predict_random_forest
from perd_perturbation.vendor.perd_repro.openness import compute_enhancer_openness


def test_openness_uses_max_over_strictly_contained_bins():
    enhancers = pd.DataFrame(
        {"enhancer_id": ["e1", "e2"], "chrom": ["chr1", "chr1"], "start": [100, 1000], "end": [500, 1100]}
    )
    # Bins 200-400 and 300-500 are contained in e1; 0-200 and 400-600 only overlap it.
    dnase = pd.DataFrame(
        {
            "chromosome": ["chr1"] * 4,
            "start": [0, 200, 300, 400],
            "end": [200, 400, 500, 600],
            "cellA": [9.0, 1.0, 2.0, 9.0],
        }
    )
    openness, n_no_bins = compute_enhancer_openness(enhancers, dnase)
    assert openness.loc["e1", "cellA"] == 2.0
    assert "e2" not in openness.index
    assert n_no_bins == 1


def test_random_forest_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    y = x[:, 0] + rng.normal(scale=0.1, size=30)
    first, _ = fit_predict_random_forest(x, y, x[:3])
    second, _ = fit_predict_random_forest(x, y, x[:3])
    np.testing.assert_array_equal(first, second)
