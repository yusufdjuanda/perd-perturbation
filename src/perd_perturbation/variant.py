"""Prediction variants for the evaluation scripts (amendment 07 D).

PERD_VARIANT unset (default): frozen PERD, file names unchanged. PERD_VARIANT=<tag>: read
perd_predictions_<tag>_<model>.parquet restricted to the frozen fitted enhancers (with the frozen meta columns, so
evaluation sets and strata are identical) and write output tables with a _<tag> suffix.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

META = ("n_genes_train", "n_genes_used", "skip_reason", "alpha")


def tag() -> str:
    return os.environ.get("PERD_VARIANT", "")


def tagged(path: Path) -> Path:
    """`path` with _<tag> inserted before the suffix (unchanged for frozen PERD)."""
    return path.with_name(f"{path.stem}_{tag()}{path.suffix}") if tag() else path


def read_predictions(directory: Path, model: str, variant: str | None = None) -> pd.DataFrame:
    """Frozen predictions, or the variant's predictions on the frozen fitted enhancers with frozen meta columns."""
    variant = tag() if variant is None else variant
    frozen = pd.read_parquet(directory / f"perd_predictions_{model}.parquet")
    if not variant:
        return frozen
    table = pd.read_parquet(directory / f"perd_predictions_{variant}_{model}.parquet")
    fitted = frozen.index[frozen["skip_reason"] == ""]
    samples = [c for c in frozen.columns if c not in META]
    values = table.reindex(fitted)[samples]
    if values.isna().any().any():
        raise ValueError(f"{variant}/{model}: missing predictions for frozen fitted enhancers")
    out = frozen.copy()
    out.loc[fitted, samples] = values
    return out
