"""PERD inference on DBKERO RNA profiles (runbook section 7; amendment 01 section A).

For each PERD training enhancer: predictor genes = its `usable_genes` that are present
in the DBKERO RNA table (original order kept); skip if fewer than 2 remain. Fit on all
167 ENCODE cells, then predict every DBKERO profile passed in. Models are not kept.

- PERD-RF: vendored `fit_predict_random_forest` (500 trees, max_features = p // 3, seed 42).
- PERD-ridge: vendored `fit_predict_one_enhancer` (StandardScaler -> RidgeCV, LOO alpha).
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from perd_perturbation.vendor.perd_repro.models import fit_predict_random_forest
from perd_perturbation.vendor.perd_repro.netopen_model import fit_predict_one_enhancer

MIN_GENES = 2
MODELS = ("rf", "ridge")


def iter_items(
    training: pd.DataFrame,
    encode_rna: pd.DataFrame,
    openness: pd.DataFrame,
    profiles: pd.DataFrame,
) -> Iterator[tuple]:
    """Yield one small, picklable work item per training enhancer (lazily, to bound memory).

    training: enhancer_id, usable_genes; encode_rna: genes x 167 cells; openness: enhancers x
    the same cells; profiles: genes x DBKERO profiles (columns are the profile names).
    """
    cells = list(encode_rna.columns)
    available = set(profiles.index)
    for enhancer_id, usable in zip(training["enhancer_id"], training["usable_genes"]):
        usable = list(usable)
        genes = [g for g in usable if g in available]
        if len(genes) < MIN_GENES:
            yield (enhancer_id, len(usable), len(genes), None, None, None)
            continue
        x_train = encode_rna.loc[genes, cells].to_numpy().T
        y_train = openness.loc[enhancer_id, cells].to_numpy()
        x_test = profiles.loc[genes].to_numpy().T
        yield (enhancer_id, len(usable), len(genes), x_train, y_train, x_test)


def predict_item(item: tuple, model: str = "rf") -> tuple:
    """Returns (enhancer_id, n_train, n_used, predictions or None, skip_reason, ridge alpha or NaN)."""
    enhancer_id, n_train, n_used, x_train, y_train, x_test = item
    if x_train is None:
        return enhancer_id, n_train, n_used, None, f"fewer_than_{MIN_GENES}_genes_in_dbkero", np.nan
    if model == "rf":
        prediction, _ = fit_predict_random_forest(x_train, y_train, x_test)
        return enhancer_id, n_train, n_used, prediction, "", np.nan
    if model == "ridge":
        prediction, alpha = fit_predict_one_enhancer(x_train, y_train, x_test)
        return enhancer_id, n_train, n_used, prediction, "", float(alpha)
    raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")


def run_perd_model(
    training: pd.DataFrame,
    encode_rna: pd.DataFrame,
    openness: pd.DataFrame,
    profiles: pd.DataFrame,
    model: str = "rf",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Predictions indexed by enhancer_id: one column per profile, plus n_genes_train,
    n_genes_used, skip_reason ('' when fitted), and (ridge only) alpha. Row order follows `training`."""
    items = iter_items(training, encode_rna, openness, profiles)
    if n_jobs == 1:
        results = [predict_item(item, model) for item in items]
    else:
        results = Parallel(n_jobs=n_jobs, batch_size=64, pre_dispatch="4*n_jobs")(
            delayed(predict_item)(item, model) for item in items
        )
    columns = list(profiles.columns)
    values = np.full((len(results), len(columns)), np.nan)
    for i, result in enumerate(results):
        if result[3] is not None:
            values[i] = result[3]
    table = pd.DataFrame(values, columns=columns, index=pd.Index([r[0] for r in results], name="enhancer_id"))
    table["n_genes_train"] = [r[1] for r in results]
    table["n_genes_used"] = [r[2] for r in results]
    table["skip_reason"] = [r[4] for r in results]
    if model == "ridge":
        table["alpha"] = [r[5] for r in results]
    return table


def run_perd_rf(training, encode_rna, openness, profiles, n_jobs: int = 1) -> pd.DataFrame:
    """PERD-RF (section 7 entry point, unchanged behavior)."""
    return run_perd_model(training, encode_rna, openness, profiles, model="rf", n_jobs=n_jobs)
