# Vendored from ../perd/src/perd_repro/models.py without modification (body below is byte-identical).
# Source repository: perd (frozen baseline commit cb36c1442cc0; transferred working tree at e169ee2).
# Source SHA-256: 83c0d2564397266f8135010a9fe0607deb599b6fe2f1c0271a7ce19559810d42 (matches data/baseline_manifest.json).
"""
Elastic net, SVR, and random forest -- the three alternatives netopen.R
never actually used, needed to test the paper's claim that random forest
outperformed the others (see README "Known reproducibility issue").

The paper's Methods don't specify hyperparameters for any of these three,
so each uses either cross-validated selection (elastic net, matching how
ridge was tuned) or a fixed, documented, reasonable default (SVR, random
forest) rather than an exhaustive search -- flagged here, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ALPHAS = np.logspace(-2, 4, 30)
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def fit_predict_elasticnet(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(alphas=ALPHAS, l1_ratio=L1_RATIOS, cv=5, max_iter=10_000),
    )
    model.fit(x_train, y_train)
    en = model.named_steps["elasticnetcv"]
    return model.predict(x_test), {"alpha": en.alpha_, "l1_ratio": en.l1_ratio_}


def fit_predict_svr(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray):
    """Fixed hyperparameters (RBF kernel, C=1.0, gamma='scale') -- no grid search, for tractability at scale."""
    model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, gamma="scale"))
    model.fit(x_train, y_train)
    return model.predict(x_test), {}


def fit_predict_random_forest(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int = 42):
    """
    Matches R's randomForest package defaults for regression, since the paper
    never overrides them: ntree=500, mtry=floor(p/3). scikit-learn's own
    RandomForestRegressor default (max_features=1.0, i.e. ALL features per
    split) is structurally different -- it's closer to plain bagging than a
    true random forest, which relies on per-split feature subsampling to
    decorrelate trees. Trees don't need feature scaling.
    """
    n_features = x_train.shape[1]
    mtry = max(1, n_features // 3)
    model = RandomForestRegressor(n_estimators=500, max_features=mtry, random_state=seed, n_jobs=1)
    model.fit(x_train, y_train)
    return model.predict(x_test), {}


def _build_xy_items(training_enhancers, openness, rna, train_cells, test_cells):
    """Pre-extract small per-enhancer numpy arrays, so parallel workers never need to pickle the full rna/openness tables."""
    items = []
    for _, row in training_enhancers.iterrows():
        eid = row["enhancer_id"]
        genes = row["usable_genes"]

        x = rna.loc[genes].T
        y = openness.loc[eid]

        items.append(
            (
                eid,
                len(genes),
                x.loc[train_cells].to_numpy(),
                y.loc[train_cells].to_numpy(),
                x.loc[test_cells].to_numpy(),
                y.loc[test_cells].to_numpy(),
            )
        )
    return items


def _fit_one_item(item, fit_predict_fn):
    eid, n_genes, x_train, y_train, x_test, y_test = item
    pred, extra = fit_predict_fn(x_train, y_train, x_test)
    return {"enhancer_id": eid, "n_genes": n_genes, "y_test": y_test, "y_pred": pred, **extra}


def run_netopen_model(
    training_enhancers: pd.DataFrame,
    openness: pd.DataFrame,
    rna: pd.DataFrame,
    train_cells: list[str],
    test_cells: list[str],
    fit_predict_fn,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Generic per-enhancer model runner -- same X/y assembly as run_netopen_ridge, any fit_predict_fn.

    n_jobs > 1 parallelizes across enhancers (not within a single fit) via joblib --
    each of the ~120k model fits is independent and tiny, so this is the effective
    place to spend extra cores, not scikit-learn's own internal n_jobs.
    """
    items = _build_xy_items(training_enhancers, openness, rna, train_cells, test_cells)

    if n_jobs == 1:
        rows = [_fit_one_item(item, fit_predict_fn) for item in items]
    else:
        rows = Parallel(n_jobs=n_jobs)(delayed(_fit_one_item)(item, fit_predict_fn) for item in items)

    return pd.DataFrame(rows)
