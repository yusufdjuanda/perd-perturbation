# Vendored from ../perd/src/perd_repro/cell_types.py without modification (body below is byte-identical).
# Source repository: perd (frozen baseline commit cb36c1442cc0; transferred working tree at e169ee2).
# Source SHA-256: 11d21e5a951c48d57bec71c20a4631b3611872898141cff16792fb851c20750c (not in data/baseline_manifest.json; hash recorded at vendoring time, 2026-10-04).
"""
Group ENCODE's 167 individual cell-line sample names into broader cell
*types* (e.g. the 6 "adrenal_gland_*" donor/age variants all collapse to
"adrenal_gland"). Needed for the paper's "DH specificity" statistic
(number of distinct cell TYPES with signal, as opposed to "DH spread",
which counts individual cell LINES) -- see README "CMap candidate
enhancer filter" for the full citation and the paper's exact definitions.

The paper doesn't publish an explicit ENCODE-sample -> cell-type mapping,
so this is our own interpretation: strip donor/age/sex/treatment metadata
from each sample's full descriptor via regex, leaving the base tissue/
cell-line name. Verified to produce 75 sensible groups from 167 samples
(e.g. all adrenal_gland donor variants collapse together; distinct types
like CD4 vs CD8 T cells stay separate).
"""

from __future__ import annotations

import re


def extract_cell_type(sample_name: str) -> str:
    full = sample_name.split("-")[0]  # first '-'-delimited field is the full descriptor
    t = re.sub(r"_(male|female)(_|$)", "_", full)
    t = re.sub(r"_(adult|embryo|child|newborn|fetal)(_\(?[\d.]+[_ ]?(years?|days?|weeks?|months?)\)?)?", "", t)
    t = re.sub(r"_\(?\d+[\d._]*\s*(years?|days?|weeks?|months?)\)?", "", t)
    t = re.sub(r"_originated_from.*", "", t)
    t = re.sub(r"_treated_with.*", "", t)
    return t.strip("_")
