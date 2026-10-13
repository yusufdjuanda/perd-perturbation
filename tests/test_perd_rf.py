import numpy as np
import pandas as pd

from perd_perturbation.perd_rf import run_perd_rf


def _fixture():
    rng = np.random.default_rng(0)
    cells = [f"cell{i}" for i in range(40)]
    genes = ["A", "B", "C", "D"]
    encode = pd.DataFrame(rng.uniform(0, 8, size=(4, 40)), index=genes, columns=cells)
    # Enhancer e1 opens with gene A; e2 has only one gene available in DBKERO.
    openness = pd.DataFrame(
        {cell: [encode.loc["A", cell] + rng.normal(scale=0.1), 1.0] for cell in cells}, index=["e1", "e2"]
    )
    training = pd.DataFrame({"enhancer_id": ["e1", "e2"], "usable_genes": [["A", "B", "C"], ["C", "D"]]})
    profiles = pd.DataFrame(
        {"cond|control": [1.0, 4.0, 4.0], "cond|treated": [7.0, 4.0, 4.0]}, index=["A", "B", "C"]
    )
    return training, encode, openness, profiles


def test_predict_absolute_then_subtract_sign_follows_driver_gene():
    training, encode, openness, profiles = _fixture()
    table = run_perd_rf(training, encode, openness, profiles)
    delta = table.loc["e1", "cond|treated"] - table.loc["e1", "cond|control"]
    assert delta > 3  # gene A rises from 1 to 7, so predicted openness must rise
    assert table.loc["e1", "n_genes_train"] == 3
    assert table.loc["e1", "n_genes_used"] == 3
    assert table.loc["e1", "skip_reason"] == ""


def test_skips_enhancers_with_fewer_than_two_dbkero_genes():
    training, encode, openness, profiles = _fixture()
    table = run_perd_rf(training, encode, openness, profiles)
    assert table.loc["e2", "n_genes_used"] == 1
    assert table.loc["e2", "skip_reason"] == "fewer_than_2_genes_in_dbkero"
    assert np.isnan(table.loc["e2", "cond|treated"])


def test_parallel_matches_serial_exactly():
    training, encode, openness, profiles = _fixture()
    serial = run_perd_rf(training, encode, openness, profiles, n_jobs=1)
    parallel = run_perd_rf(training, encode, openness, profiles, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_ridge_sign_skip_and_alpha():
    from perd_perturbation.perd_rf import run_perd_model

    training, encode, openness, profiles = _fixture()
    table = run_perd_model(training, encode, openness, profiles, model="ridge")
    assert table.loc["e1", "cond|treated"] - table.loc["e1", "cond|control"] > 3
    assert table.loc["e1", "alpha"] > 0
    assert table.loc["e2", "skip_reason"] == "fewer_than_2_genes_in_dbkero"
    assert np.isnan(table.loc["e2", "alpha"])
