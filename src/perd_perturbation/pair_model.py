"""Perturbation-pair model (amendment 10, roadmap E4): conditions and features.

A condition = one measured perturbation (treated vs control samples of accessibility and RNA). Features per
(enhancer, condition), in a fixed column order:
  [arch:<a> = motif-density z_a(e) * delta archetype activity_a(c)]  (archetypes with activity)
  [chip:<k> = bound_k(e) * delta TF activity_k(c)]                   (scored ChIP-binding TFs)
  tg_mean, tg_max, tg_min   (delta log2 FPKM over e's target genes)
  baseline                  (measured control openness of e in c)
  prior_mean, prior_sd, length_kb, n_targets, n_bound               (static)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

STATIC = ["prior_mean", "prior_sd", "length_kb", "n_targets", "n_bound"]


@dataclass
class Condition:
    dataset: str
    unit: str
    name: str
    acc_control: list[str]
    acc_treated: list[str]
    rna_control: list[str]
    rna_treated: list[str]
    threshold: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class EnhancerTables:
    enhancers: pd.Index
    arch_z: pd.DataFrame          # enhancers x archetypes (motif density z)
    bound: pd.DataFrame           # enhancers x scored binding TFs (0/1)
    targets: sparse.csr_matrix    # enhancers x reference genes
    genes: pd.Index               # reference genes
    static: pd.DataFrame          # enhancers x STATIC
    arch_tfs: dict[str, list[str]]  # archetype -> scored TFs

    def feature_names(self) -> list[str]:
        return ([f"arch:{a}" for a in self.arch_z.columns] + [f"chip:{k}" for k in self.bound.columns]
                + ["tg_mean", "tg_max", "tg_min", "baseline"] + STATIC)


def _row_reduce(matrix: sparse.csr_matrix, values: np.ndarray, op) -> np.ndarray:
    out = np.zeros(matrix.shape[0], dtype=np.float32)
    nonempty = np.diff(matrix.indptr) > 0
    if matrix.nnz:
        out[nonempty] = op.reduceat(values[matrix.indices], matrix.indptr[:-1][nonempty])
    return out


def delta_activity(activity: pd.DataFrame, cond: Condition) -> pd.Series:
    return activity[cond.rna_treated].mean(axis=1) - activity[cond.rna_control].mean(axis=1)


def condition_features(t: EnhancerTables, d_act: pd.Series, d_rna: pd.Series, baseline: pd.Series) -> np.ndarray:
    """Feature matrix (enhancers x features, float32) for one condition."""
    d_arch = np.array([d_act.reindex(t.arch_tfs[a]).mean() for a in t.arch_z.columns], dtype=np.float32)
    d_chip = d_act.reindex(t.bound.columns).fillna(0.0).to_numpy(dtype=np.float32)
    dr = d_rna.reindex(t.genes).fillna(0.0).to_numpy(dtype=np.float32)
    n_t = np.maximum(np.diff(t.targets.indptr), 1).astype(np.float32)
    tg_mean = (t.targets @ dr) / n_t
    blocks = [t.arch_z.to_numpy(dtype=np.float32) * np.nan_to_num(d_arch)[None, :],
              t.bound.to_numpy(dtype=np.float32) * d_chip[None, :],
              np.column_stack([tg_mean, _row_reduce(t.targets, dr, np.maximum), _row_reduce(t.targets, dr, np.minimum)]),
              baseline.reindex(t.enhancers).to_numpy(dtype=np.float32)[:, None],
              t.static.loc[t.enhancers, STATIC].to_numpy(dtype=np.float32)]
    return np.hstack(blocks).astype(np.float32)


def unit_weights(conditions: list[Condition]) -> dict[str, float]:
    """Per-condition weight so that every unit has total weight 1."""
    per_unit = pd.Series([c.unit for c in conditions]).value_counts()
    return {c.name: 1.0 / per_unit[c.unit] for c in conditions}
