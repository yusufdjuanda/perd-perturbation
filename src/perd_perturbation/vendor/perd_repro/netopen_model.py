# Vendored from ../perd/src/perd_repro/netopen_model.py without modification (body below is byte-identical).
# Source repository: perd (frozen baseline commit cb36c1442cc0; transferred working tree at e169ee2).
# Source SHA-256: d28f9b9fc45207ad5098422de5fb5803123fa37a809ed7c7236d6070ccd97c96 (not in data/baseline_manifest.json; hash recorded at vendoring time, 2026-10-02).
"""
Per-enhancer ridge regression, translated from netopen.R:

    cv.Ri <- cv.glmnet(Xrn, Yrn, alpha = 0)
    model.Ri <- glmnet(Xrn, Yrn, alpha = 0, lambda = cv.Ri$lambda.min)
    pre <- predict(model.Ri, newx = Xst)

Implementation note (documented approximation, not a method change): glmnet
standardizes predictors by default before fitting ridge, and its lambda path
is parameterized differently from scikit-learn's `alpha` (glmnet's objective
is scaled by 1/(2n); sklearn's isn't) -- so this will not reproduce glmnet's
exact lambda values or coefficients. We match the *behavior* instead:
standardize features, then pick the penalty strength by cross-validation.
scikit-learn's `RidgeCV(cv=None)` uses an efficient closed-form leave-one-out
CV, which also matches the paper's own stated "leave-one-out cross-validation"
wording more closely than glmnet's actual 10-fold CV default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-2, 6, 50)


def fit_predict_one_enhancer(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, float]:
    """Fit standardized ridge with LOO-CV-selected alpha; return (predictions, chosen alpha)."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS, cv=None))
    model.fit(x_train, y_train)
    chosen_alpha = model.named_steps["ridgecv"].alpha_
    return model.predict(x_test), chosen_alpha


def run_netopen_ridge(
    training_enhancers: pd.DataFrame,
    openness: pd.DataFrame,
    rna: pd.DataFrame,
    train_cells: list[str],
    test_cells: list[str],
) -> pd.DataFrame:
    """
    training_enhancers: columns enhancer_id, usable_genes (from step 08)
    openness: enhancer_id-indexed, cell columns (from step 04)
    rna: gene_symbol-indexed, cell columns (from step 03)

    Returns one row per enhancer: enhancer_id, chosen alpha, predicted
    openness for every test cell, and the actual test-cell openness
    (for downstream metric computation).
    """
    rows = []
    for _, row in training_enhancers.iterrows():
        eid = row["enhancer_id"]
        genes = row["usable_genes"]

        x = rna.loc[genes].T  # cells x genes
        y = openness.loc[eid]

        x_train = x.loc[train_cells].to_numpy()
        x_test = x.loc[test_cells].to_numpy()
        y_train = y.loc[train_cells].to_numpy()
        y_test = y.loc[test_cells].to_numpy()

        pred, alpha = fit_predict_one_enhancer(x_train, y_train, x_test)

        rows.append(
            {
                "enhancer_id": eid,
                "n_genes": len(genes),
                "alpha": alpha,
                "y_test": y_test,
                "y_pred": pred,
            }
        )

    return pd.DataFrame(rows)
