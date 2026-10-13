"""Shared (across-enhancer) model features for amendment 08.

One row per (enhancer, cell). 446 columns in a fixed order:
  0 prior_mean, 1 prior_sd, 2 n_targets, 3 n_bound        (static per enhancer)
  4 tg_mean, 5 tg_max, 6 tg_min                           (target-gene RNA z in the cell)
  7 .. 7+K-1            bound_k                           (static, K = binding TFs)
  7+K .. 7+2K-1         bound_k * RNA z of TF k in the cell
  7+2K .. 7+2K+A-1      bound_k * activity of TF k        (A = binding TFs with an activity score)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Static:
    enhancers: pd.Index
    genes: pd.Index            # reference genes (row order of the RNA z table)
    tfs: list[str]             # binding TFs (K)
    scored: list[str]          # binding TFs with activity (A), a subset of tfs
    targets: sparse.csr_matrix  # enhancers x genes, 1 where gene is a present target
    bound: np.ndarray          # enhancers x K, float32 0/1
    n_targets: np.ndarray
    n_bound: np.ndarray

    @property
    def n_features(self) -> int:
        return 7 + 2 * len(self.tfs) + len(self.scored)

    def feature_names(self) -> list[str]:
        return (["prior_mean", "prior_sd", "n_targets", "n_bound", "tg_mean", "tg_max", "tg_min"]
                + [f"bound:{k}" for k in self.tfs] + [f"bound_x_rna:{k}" for k in self.tfs] + [f"bound_x_act:{k}" for k in self.scored])

    def static_columns(self) -> np.ndarray:
        """Indices of columns that do not depend on the cell."""
        k = len(self.tfs)
        return np.r_[0:4, 7:7 + k]


def build_static(enhancers, targets: pd.Series, binding: pd.Series, genes: pd.Index, scored_tfs: set[str]) -> Static:
    enhancers = pd.Index(enhancers)
    tfs = sorted({tf for e in enhancers for tf in (binding.get(e) if binding.get(e) is not None else [])})
    scored = [tf for tf in tfs if tf in scored_tfs]
    gene_pos = {g: i for i, g in enumerate(genes)}
    tf_pos = {tf: i for i, tf in enumerate(tfs)}
    rows, cols = [], []
    bound = np.zeros((len(enhancers), len(tfs)), dtype=np.float32)
    for r, e in enumerate(enhancers):
        t = targets.get(e)
        for g in (t if t is not None else []):
            if g in gene_pos:
                rows.append(r)
                cols.append(gene_pos[g])
        b = binding.get(e)
        for tf in (b if b is not None else []):
            bound[r, tf_pos[tf]] = 1.0
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(enhancers), len(genes)))
    matrix.sum_duplicates()
    matrix.data[:] = 1.0
    n_targets = np.diff(matrix.indptr).astype(np.float32)
    return Static(enhancers, pd.Index(genes), tfs, scored, matrix, bound, n_targets, bound.sum(axis=1))


def _row_reduce(matrix: sparse.csr_matrix, values: np.ndarray, op) -> np.ndarray:
    out = np.zeros(matrix.shape[0], dtype=np.float32)
    nonempty = np.diff(matrix.indptr) > 0
    if matrix.nnz:
        reduced = op.reduceat(values[matrix.indices], matrix.indptr[:-1][nonempty])
        out[nonempty] = reduced
    return out


def cell_features(static: Static, prior_mean: np.ndarray, prior_sd: np.ndarray, z: pd.Series, activity: pd.Series) -> np.ndarray:
    """Features (enhancers x n_features, float32) for one cell. z: RNA z over static.genes; activity: TF activity."""
    zv = z.reindex(static.genes).fillna(0.0).to_numpy(dtype=np.float32)
    n_t = np.maximum(static.n_targets, 1.0)
    tg_mean = (static.targets @ zv) / n_t
    tg_max = _row_reduce(static.targets, zv, np.maximum)
    tg_min = _row_reduce(static.targets, zv, np.minimum)
    tf_rna = z.reindex(static.tfs).fillna(0.0).to_numpy(dtype=np.float32)
    tf_act = activity.reindex(static.scored).fillna(0.0).to_numpy(dtype=np.float32)
    scored_idx = [static.tfs.index(tf) for tf in static.scored]
    out = np.empty((len(static.enhancers), static.n_features), dtype=np.float32)
    k = len(static.tfs)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = prior_mean, prior_sd, static.n_targets, static.n_bound
    out[:, 4], out[:, 5], out[:, 6] = tg_mean, tg_max, tg_min
    out[:, 7:7 + k] = static.bound
    out[:, 7 + k:7 + 2 * k] = static.bound * tf_rna
    out[:, 7 + 2 * k:] = static.bound[:, scored_idx] * tf_act
    return out


def pair_features(static: Static, features_i: np.ndarray, features_j: np.ndarray) -> np.ndarray:
    """Difference features for a pair: cell-dependent columns i - j, static columns kept from i."""
    out = features_i - features_j
    cols = static.static_columns()
    out[:, cols] = features_i[:, cols]
    return out
