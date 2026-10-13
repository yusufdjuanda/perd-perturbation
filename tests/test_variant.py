from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perd_perturbation import variant


def write(directory: Path, name: str, table: pd.DataFrame) -> None:
    table.to_parquet(directory / name)


def test_tagged_and_read_predictions(tmp_path, monkeypatch):
    frozen = pd.DataFrame({"s1": [1.0, np.nan], "n_genes_used": [3, 1], "skip_reason": ["", "few"]}, index=pd.Index(["e1", "e2"], name="enhancer_id"))
    tfadd = pd.DataFrame({"s1": [9.0, 7.0], "n_genes_used": [5, 2], "skip_reason": ["", ""]}, index=pd.Index(["e1", "e2"], name="enhancer_id"))
    write(tmp_path, "perd_predictions_rf.parquet", frozen)
    write(tmp_path, "perd_predictions_tfadd_rf.parquet", tfadd)

    monkeypatch.delenv("PERD_VARIANT", raising=False)
    assert variant.tagged(Path("t/x.csv")) == Path("t/x.csv")
    assert variant.read_predictions(tmp_path, "rf").loc["e1", "s1"] == 1.0

    monkeypatch.setenv("PERD_VARIANT", "tfadd")
    assert variant.tagged(Path("t/x.csv")) == Path("t/x_tfadd.csv")
    out = variant.read_predictions(tmp_path, "rf")
    assert out.loc["e1", "s1"] == 9.0
    assert np.isnan(out.loc["e2", "s1"])  # not in the frozen fitted set
    assert out.loc["e1", "n_genes_used"] == 3  # frozen meta kept


def test_read_predictions_requires_all_frozen_fitted(tmp_path, monkeypatch):
    frozen = pd.DataFrame({"s1": [1.0], "skip_reason": [""]}, index=pd.Index(["e1"], name="enhancer_id"))
    broken = pd.DataFrame({"s1": [np.nan], "skip_reason": [""]}, index=pd.Index(["e1"], name="enhancer_id"))
    write(tmp_path, "perd_predictions_rf.parquet", frozen)
    write(tmp_path, "perd_predictions_tfrep_rf.parquet", broken)
    monkeypatch.setenv("PERD_VARIANT", "tfrep")
    with pytest.raises(ValueError):
        variant.read_predictions(tmp_path, "rf")
