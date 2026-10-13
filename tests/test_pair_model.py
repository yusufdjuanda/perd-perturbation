import numpy as np
import pandas as pd
from scipy import sparse

from perd_perturbation import pair_model


def tables():
    enh = pd.Index(["e1", "e2"])
    genes = pd.Index(["G1", "G2", "G3"])
    targets = sparse.csr_matrix(np.array([[1, 1, 0], [0, 0, 0]], dtype=np.float32))
    static = pd.DataFrame({"prior_mean": [1.0, 2.0], "prior_sd": [0.1, 0.2], "length_kb": [1.0, 0.5], "n_targets": [2, 0], "n_bound": [1, 0]}, index=enh)
    return pair_model.EnhancerTables(enh, pd.DataFrame({"NR/20": [2.0, -0.5]}, index=enh), pd.DataFrame({"NR3C1": [1.0, 0.0]}, index=enh),
                                     targets, genes, static, {"NR/20": ["NR3C1", "AR"]})


def test_condition_features_layout_and_values():
    t = tables()
    d_act = pd.Series({"NR3C1": 2.0, "AR": 0.0})
    d_rna = pd.Series({"G1": 1.0, "G2": 3.0})
    x = pair_model.condition_features(t, d_act, d_rna, pd.Series({"e1": 0.5, "e2": 1.5}))
    names = t.feature_names()
    assert x.shape == (2, len(names))
    row = dict(zip(names, x[0]))
    assert row["arch:NR/20"] == 2.0 * 1.0  # z 2.0 x mean(dGR 2, dAR 0)
    assert row["chip:NR3C1"] == 2.0
    assert row["tg_mean"] == 2.0 and row["tg_max"] == 3.0 and row["tg_min"] == 1.0
    assert row["baseline"] == 0.5 and row["prior_mean"] == 1.0
    row2 = dict(zip(names, x[1]))
    assert row2["arch:NR/20"] == -0.5 and row2["chip:NR3C1"] == 0.0 and row2["tg_mean"] == 0.0


def test_unit_weights_sum_to_one_per_unit():
    c = [pair_model.Condition("d", u, f"{u}{i}", [], [], [], []) for u, n in (("A", 11), ("B", 1)) for i in range(n)]
    w = pair_model.unit_weights(c)
    assert abs(sum(v for k, v in w.items() if k.startswith("A")) - 1) < 1e-12
    assert w["B0"] == 1.0
