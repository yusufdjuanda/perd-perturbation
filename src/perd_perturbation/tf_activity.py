"""TF activity from RNA (amendment 07 A) and the TF-activity feature lists for PERD (amendment 07 C).

Activity = univariate linear model (ULM, as in decoupleR): per TF and sample, regress the sample's gene z-scores on
the TF's signed target weights (0 for non-targets) and take the t-value of the slope. Gene z-scores use the ENCODE
167-cell training table as the fixed reference, so test data never shape the scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVITY_PREFIX = "act:"
MIN_TARGETS = 5


def network_from_collectri(raw: pd.DataFrame) -> pd.DataFrame:
    """OmniPath CollecTRI rows -> (source, target, weight) with complexes expanded and duplicates resolved.

    weight = -1 if consensus_inhibition and not consensus_stimulation, else +1. A complex source (`COMPLEX:...`,
    gene symbols joined by '_') gives its edges to each member gene. Duplicate pairs: sign of the summed weights;
    pairs summing to 0 are dropped.
    """
    inhibition = raw["consensus_inhibition"].astype(str).str.lower().eq("true")
    stimulation = raw["consensus_stimulation"].astype(str).str.lower().eq("true")
    weight = np.where(inhibition & ~stimulation, -1, 1)
    edges = pd.DataFrame({"source": raw["source_genesymbol"].astype(str), "target": raw["target_genesymbol"].astype(str), "weight": weight})
    is_complex = raw["source"].astype(str).str.startswith("COMPLEX:").to_numpy()
    expanded = edges[is_complex].assign(source=lambda d: d["source"].str.split("_")).explode("source")
    edges = pd.concat([edges[~is_complex], expanded], ignore_index=True)
    summed = edges.groupby(["source", "target"], as_index=False)["weight"].sum()
    summed = summed[summed["weight"] != 0]
    summed["weight"] = np.sign(summed["weight"]).astype(int)
    return summed.reset_index(drop=True)


def reference_scale(encode_rna: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-gene mean and SD across the training cells; genes with SD = 0 are dropped."""
    mean = encode_rna.mean(axis=1)
    sd = encode_rna.std(axis=1, ddof=1)
    keep = sd > 0
    return mean[keep], sd[keep]


def standardize(profiles: pd.DataFrame, mean: pd.Series, sd: pd.Series) -> pd.DataFrame:
    """z = (value - reference mean) / reference SD on the reference genes; genes missing from `profiles` get 0."""
    values = profiles.reindex(mean.index)
    z = values.sub(mean, axis=0).div(sd, axis=0)
    return z.fillna(0.0)


def weight_matrix(network: pd.DataFrame, genes: pd.Index, min_targets: int = MIN_TARGETS) -> pd.DataFrame:
    """Genes x TFs signed weights (0 = not a target), keeping TFs with >= min_targets targets among `genes`."""
    edges = network[network["target"].isin(genes)]
    counts = edges.groupby("source")["target"].nunique()
    tfs = counts.index[counts >= min_targets]
    edges = edges[edges["source"].isin(tfs)]
    matrix = edges.pivot_table(index="target", columns="source", values="weight", aggfunc="sum", fill_value=0)
    return matrix.reindex(index=genes, fill_value=0).reindex(columns=sorted(tfs)).astype(float)


def ulm(z: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """TFs x samples t-values of the slope b in z_gene = a + b * w_gene (one fit per TF and sample).

    Closed form: with r the Pearson correlation of w and z over genes and n genes, t = r * sqrt((n - 2) / (1 - r^2)).
    """
    if not z.index.equals(weights.index):
        raise ValueError("z and weights must share the same gene index in the same order")
    n = z.shape[0]
    w = weights.to_numpy()
    y = z.to_numpy()
    wc = w - w.mean(axis=0)
    yc = y - y.mean(axis=0)
    denom = np.outer(np.linalg.norm(wc, axis=0), np.linalg.norm(yc, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (wc.T @ yc) / denom
        t = r * np.sqrt((n - 2) / np.clip(1 - r**2, 1e-12, None))
    return pd.DataFrame(t, index=weights.columns, columns=z.columns)


def activity(profiles: pd.DataFrame, mean: pd.Series, sd: pd.Series, weights: pd.DataFrame) -> pd.DataFrame:
    """TF activity (TFs x samples) for RNA profiles (genes x samples) on the fixed reference scale."""
    return ulm(standardize(profiles, mean, sd), weights)


def feature_lists(
    training: pd.DataFrame,
    binding: pd.Series,
    targets: pd.Series,
    scored_tfs: set[str],
    variant: str,
) -> pd.DataFrame:
    """Per-enhancer feature lists for the amendment 07 variants.

    training: enhancer_id, usable_genes (frozen PERD); binding: enhancer_id -> binding TFs; targets: enhancer_id ->
    GeneHancer target genes; scored_tfs: TFs with an activity score.
    - 'tfadd': usable_genes plus act:TF for each binding TF with a score.
    - 'tfrep': as 'tfadd', but the RNA of each scored binding TF is removed unless that TF is also a target gene.
    """
    if variant not in ("tfadd", "tfrep"):
        raise ValueError(f"unknown variant {variant!r}")
    rows = []
    for enhancer_id, usable in zip(training["enhancer_id"], training["usable_genes"]):
        bound = binding.get(enhancer_id, [])
        bound = [] if bound is None else list(bound)
        scored = sorted(set(bound) & scored_tfs)
        genes = list(usable)
        if variant == "tfrep":
            target_set = set(targets.get(enhancer_id, []) if targets.get(enhancer_id) is not None else [])
            drop = {tf for tf in scored if tf not in target_set}
            genes = [g for g in genes if g not in drop]
        rows.append({"enhancer_id": enhancer_id, "usable_genes": genes + [ACTIVITY_PREFIX + tf for tf in scored], "n_activity": len(scored)})
    return pd.DataFrame(rows)


def with_activity(rna: pd.DataFrame, activity_table: pd.DataFrame) -> pd.DataFrame:
    """Stack RNA (genes x samples) and activity (TFs x samples, rows renamed act:TF) into one feature table."""
    act = activity_table.rename(index=lambda tf: ACTIVITY_PREFIX + tf)
    return pd.concat([rna, act[rna.columns]], axis=0)
