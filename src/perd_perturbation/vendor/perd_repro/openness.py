# Vendored from ../perd/src/perd_repro/openness.py without modification (body below is byte-identical).
# Source repository: perd (frozen baseline commit cb36c1442cc0; transferred working tree at e169ee2).
# Source SHA-256: 5d6d8ffdc5b554a289a7967bfe5a80bbd029da7a5ac20a70fa6ed4a03b7bbb13 (not in data/baseline_manifest.json; hash recorded at vendoring time).
"""
Compute enhancer openness: for each GeneHancer enhancer, the max DNase signal
among all DNase bins falling *entirely inside* its interval, per cell.

Translated from the original PERD README's R loop:
    m <- DNase_data[enhancer.withOpen.lab1[[i]], ]
    ... apply(m, 2, max)
where `enhancer.withOpen.lab1[[i]]` is the set of DNase bins with
`start >= enhancer.start & end <= enhancer.end` (strict containment, not
just any overlap).
"""

from __future__ import annotations

import pandas as pd
import pyranges as pr


def compute_enhancer_openness(enhancers: pd.DataFrame, dnase: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    enhancers: columns enhancer_id, chrom, start, end
    dnase: columns chromosome, start, end, <167 per-cell signal columns>

    Returns (openness_df, n_enhancers_with_no_bins) where openness_df is
    indexed by enhancer_id with one column per cell (same columns as `dnase`).
    """
    signal_cols = [c for c in dnase.columns if c not in ("chromosome", "start", "end")]

    enh_pr = pr.PyRanges(
        enhancers.rename(columns={"chrom": "Chromosome", "start": "Start", "end": "End"})[
            ["Chromosome", "Start", "End", "enhancer_id"]
        ]
    )
    dnase_pr = pr.PyRanges(
        dnase.rename(columns={"chromosome": "Chromosome", "start": "Start", "end": "End"})[
            ["Chromosome", "Start", "End"] + signal_cols
        ]
    )

    joined = enh_pr.join(dnase_pr).df

    # keep only bins strictly contained within the enhancer interval (Start/End
    # here are the enhancer's; Start_b/End_b are the DNase bin's)
    contained = joined[(joined["Start_b"] >= joined["Start"]) & (joined["End_b"] <= joined["End"])]

    openness = contained.groupby("enhancer_id")[signal_cols].max()

    n_no_bins = enhancers["enhancer_id"].nunique() - openness.index.nunique()
    return openness, n_no_bins
