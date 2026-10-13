import numpy as np
import pandas as pd
import pytest

from perd_perturbation import shared_model


def make_static():
    enhancers = ["e1", "e2", "e3"]
    targets = pd.Series({"e1": ["G1", "G2", "MISSING"], "e2": [], "e3": ["G3"]})
    binding = pd.Series({"e1": ["NR3C1", "JUN"], "e2": ["JUN"], "e3": None})
    genes = pd.Index(["G1", "G2", "G3", "NR3C1", "JUN"])
    return shared_model.build_static(enhancers, targets, binding, genes, scored_tfs={"NR3C1"})


def test_static_layout():
    s = make_static()
    assert s.tfs == ["JUN", "NR3C1"] and s.scored == ["NR3C1"]
    assert s.n_features == 7 + 2 * 2 + 1
    assert list(s.n_targets) == [2, 0, 1]
    assert list(s.n_bound) == [2, 1, 0]
    assert len(s.feature_names()) == s.n_features
    assert list(s.static_columns()) == [0, 1, 2, 3, 7, 8]


def test_cell_features_values():
    s = make_static()
    z = pd.Series({"G1": 1.0, "G2": 3.0, "G3": -2.0, "NR3C1": 0.5, "JUN": -1.0})
    act = pd.Series({"NR3C1": 2.5, "OTHER": 9.0})
    f = shared_model.cell_features(s, np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]), z, act)
    names = s.feature_names()
    row = dict(zip(names, f[0]))
    assert row["tg_mean"] == pytest.approx(2.0) and row["tg_max"] == 3.0 and row["tg_min"] == 1.0
    assert row["bound:JUN"] == 1 and row["bound_x_rna:JUN"] == -1.0 and row["bound_x_rna:NR3C1"] == 0.5
    assert row["bound_x_act:NR3C1"] == 2.5
    row2 = dict(zip(names, f[1]))
    assert row2["tg_mean"] == 0.0 and row2["tg_max"] == 0.0  # no present targets
    assert row2["bound_x_act:NR3C1"] == 0.0  # not bound
    assert row2["bound_x_rna:JUN"] == -1.0


def test_pair_features_keep_static():
    s = make_static()
    zi = pd.Series({"G1": 1.0, "G2": 3.0, "G3": -2.0, "NR3C1": 0.5, "JUN": -1.0})
    zj = zi * 0
    act = pd.Series({"NR3C1": 2.5})
    prior = np.array([1.0, 2.0, 3.0])
    fi = shared_model.cell_features(s, prior, prior, zi, act)
    fj = shared_model.cell_features(s, prior, prior, zj, act * 0)
    d = shared_model.pair_features(s, fi, fj)
    assert np.array_equal(d[:, s.static_columns()], fi[:, s.static_columns()])
    assert d[0, 4] == pytest.approx(2.0)
    assert d[0, -1] == pytest.approx(2.5)
