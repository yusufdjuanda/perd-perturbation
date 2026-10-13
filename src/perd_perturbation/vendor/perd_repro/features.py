# Vendored from ../perd/src/perd_repro/features.py without modification (body below is byte-identical).
# Source repository: perd (frozen baseline commit cb36c1442cc0; transferred working tree at e169ee2).
# Source SHA-256: cc38554467a6a2a132161ee13ae821d197505898b61ebb6260bd88d4d948dceb (matches data/baseline_manifest.json).
"""
Assemble the per-enhancer predictor gene lists NetOpen trains on, applying
the original method's filter:

    Flist <- intersect(RNA_genes, union(target_genes, TF_genes))
    use_lab <- which(length(Flist) >= 2)

(from netopen.R -- see its docstring reference in scripts/08_...).
"""

from __future__ import annotations

import pandas as pd


def assemble_predictor_gene_lists(
    trainable_enhancer_ids,
    target_genes_by_id: dict[str, list[str]],
    tf_genes_by_id: dict[str, list[str]],
    available_gene_symbols: set[str],
) -> pd.DataFrame:
    """
    trainable_enhancer_ids: enhancers that have an openness value (a Y to train on)
    target_genes_by_id / tf_genes_by_id: enhancer_id -> list[gene symbol]
    available_gene_symbols: gene symbols actually present in the expression data

    Returns a DataFrame (enhancer_id, usable_genes, n_genes), restricted to
    enhancers with >=2 usable genes -- matches netopen.R's `use_lab` filter.
    """
    rows = []
    for eid in trainable_enhancer_ids:
        tg = target_genes_by_id.get(eid, [])
        tf = tf_genes_by_id.get(eid, [])
        usable = sorted((set(tg) | set(tf)) & available_gene_symbols)
        if len(usable) >= 2:
            rows.append({"enhancer_id": eid, "usable_genes": usable, "n_genes": len(usable)})

    return pd.DataFrame(rows)
