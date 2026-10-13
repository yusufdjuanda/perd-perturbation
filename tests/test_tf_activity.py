import numpy as np
import pandas as pd
import pytest

from perd_perturbation import tf_activity


def test_ulm_matches_least_squares_t_value():
    rng = np.random.default_rng(0)
    genes = pd.Index([f"g{i}" for i in range(200)])
    w = np.zeros((200, 2))
    w[:20, 0] = 1
    w[20:30, 0] = -1
    w[50:80, 1] = 1
    weights = pd.DataFrame(w, index=genes, columns=["TF1", "TF2"])
    z = pd.DataFrame(rng.standard_normal((200, 3)), index=genes, columns=["s1", "s2", "s3"])
    z.iloc[:20, 0] += 1.5
    result = tf_activity.ulm(z, weights)
    for tf in weights.columns:
        x = np.column_stack([np.ones(200), weights[tf].to_numpy()])
        for sample in z.columns:
            y = z[sample].to_numpy()
            beta, residual, _, _ = np.linalg.lstsq(x, y, rcond=None)
            sigma2 = residual[0] / (200 - 2)
            se = np.sqrt(sigma2 * np.linalg.inv(x.T @ x)[1, 1])
            assert result.loc[tf, sample] == pytest.approx(beta[1] / se, rel=1e-8)
    assert result.loc["TF1", "s1"] > 5


def test_activity_uses_reference_scale_and_fills_missing_genes():
    encode = pd.DataFrame({"c1": [0.0, 2.0, 5.0, 1.0], "c2": [2.0, 4.0, 5.0, 3.0]}, index=["A", "B", "C", "D"])
    mean, sd = tf_activity.reference_scale(encode)
    assert list(mean.index) == ["A", "B", "D"]  # C has SD 0
    profiles = pd.DataFrame({"x": [1.0, 10.0]}, index=["A", "E"])
    z = tf_activity.standardize(profiles, mean, sd)
    assert z.loc["A", "x"] == pytest.approx(0.0)
    assert z.loc["B", "x"] == 0.0 and z.loc["D", "x"] == 0.0


def test_network_expands_complexes_and_resolves_duplicates():
    raw = pd.DataFrame({
        "source": ["P1", "COMPLEX:P2_P3", "P1", "P4", "P4"],
        "source_genesymbol": ["NR3C1", "FOS_JUN", "NR3C1", "REL", "REL"],
        "target_genesymbol": ["FKBP5", "MMP1", "IL6", "T1", "T1"],
        "consensus_stimulation": [True, True, False, True, False],
        "consensus_inhibition": [False, False, True, False, True],
    })
    net = tf_activity.network_from_collectri(raw)
    pairs = {(s, t): w for s, t, w in net.itertuples(index=False)}
    assert pairs[("NR3C1", "FKBP5")] == 1
    assert pairs[("NR3C1", "IL6")] == -1
    assert pairs[("FOS", "MMP1")] == 1 and pairs[("JUN", "MMP1")] == 1
    assert ("REL", "T1") not in pairs  # +1 and -1 cancel


def test_weight_matrix_min_targets():
    net = pd.DataFrame({"source": ["A"] * 5 + ["B"] * 4, "target": [f"g{i}" for i in range(5)] + [f"g{i}" for i in range(4)], "weight": 1})
    genes = pd.Index([f"g{i}" for i in range(10)])
    m = tf_activity.weight_matrix(net, genes, min_targets=5)
    assert list(m.columns) == ["A"]
    assert m.shape == (10, 1) and m["A"].sum() == 5


def test_feature_lists_add_and_replace():
    training = pd.DataFrame({"enhancer_id": ["e1", "e2"], "usable_genes": [["GENE1", "NR3C1", "JUN", "MYC"], ["GENE2", "CTCF"]]})
    binding = pd.Series({"e1": ["NR3C1", "JUN", "MYC", "EP300"], "e2": ["CTCF"]})
    targets = pd.Series({"e1": ["GENE1", "MYC"], "e2": ["GENE2"]})
    scored = {"NR3C1", "JUN", "MYC"}
    add = tf_activity.feature_lists(training, binding, targets, scored, "tfadd").set_index("enhancer_id")
    assert add.loc["e1", "usable_genes"] == ["GENE1", "NR3C1", "JUN", "MYC", "act:JUN", "act:MYC", "act:NR3C1"]
    assert add.loc["e2", "usable_genes"] == ["GENE2", "CTCF"]
    rep = tf_activity.feature_lists(training, binding, targets, scored, "tfrep").set_index("enhancer_id")
    # MYC is also a target gene, so its RNA stays; NR3C1 and JUN RNA are replaced by activity.
    assert rep.loc["e1", "usable_genes"] == ["GENE1", "MYC", "act:JUN", "act:MYC", "act:NR3C1"]
    assert rep.loc["e1", "n_activity"] == 3


def test_with_activity_stacks_rows():
    rna = pd.DataFrame({"s1": [1.0], "s2": [2.0]}, index=["GENE1"])
    act = pd.DataFrame({"s2": [0.5], "s1": [-0.5]}, index=["NR3C1"])
    out = tf_activity.with_activity(rna, act)
    assert list(out.index) == ["GENE1", "act:NR3C1"]
    assert out.loc["act:NR3C1", "s1"] == -0.5
