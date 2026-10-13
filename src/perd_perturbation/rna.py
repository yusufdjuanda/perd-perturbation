"""Gene-level RNA quantification from salmon transcript estimates (runbook section 5b).

PERD's ENCODE RNA matrix is gene-level log2(FPKM/RPKM + 1) indexed by symbol, so
salmon transcript estimates are aggregated to genes with the tximport length
convention and converted to FPKM on the same scale.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def strip_version(ensembl_id: str) -> str:
    """'ENSG00000182378.14_PAR_Y' -> 'ENSG00000182378'; PAR_Y copies merge with their X copy."""
    return ensembl_id.split(".")[0]


def parse_gencode_header(header: str) -> dict[str, str]:
    """Parse a GENCODE transcript FASTA header (with or without the leading '>').

    Fields: transcript_id | gene_id | havana_gene | havana_transcript |
    transcript_name | gene_name | length | biotype |
    """
    fields = header.lstrip(">").strip().split("|")
    return {
        "transcript_id": fields[0],
        "gene_id": strip_version(fields[1]),
        "gene_name": fields[5],
        "gene_biotype_hint": fields[7] if len(fields) > 7 else "",
    }


def read_gencode_tx2gene(fasta_gz: Path) -> pd.DataFrame:
    """Transcript -> gene table from a GENCODE transcripts FASTA (salmon --gencode keeps field 1 as Name)."""
    rows = []
    with gzip.open(fasta_gz, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                rows.append(parse_gencode_header(line))
    return pd.DataFrame(rows)


def aggregate_to_genes(quant: pd.DataFrame, tx2gene: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a salmon quant.sf table to genes.

    quant: columns Name, Length, EffectiveLength, TPM, NumReads.
    Returns a frame indexed by unversioned Ensembl gene ID with columns
    reads, tpm, effective_length, fpkm.

    Gene effective length is the TPM-weighted mean of its transcripts' effective
    lengths (tximport); genes with zero total TPM use the unweighted mean.
    FPKM = 1e9 * gene_reads / (gene_effective_length * total_mapped_reads), where
    total_mapped_reads is the sum of NumReads over all transcripts.
    """
    mapping = tx2gene.set_index("transcript_id")["gene_id"]
    missing = ~quant["Name"].isin(mapping.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} quantified transcripts are absent from tx2gene")
    frame = quant.assign(gene_id=quant["Name"].map(mapping).to_numpy())
    frame["tpm_x_len"] = frame["TPM"] * frame["EffectiveLength"]
    grouped = frame.groupby("gene_id", sort=True)
    genes = pd.DataFrame(
        {
            "reads": grouped["NumReads"].sum(),
            "tpm": grouped["TPM"].sum(),
            "weighted_len": grouped["tpm_x_len"].sum(),
            "mean_len": grouped["EffectiveLength"].mean(),
        }
    )
    genes["effective_length"] = np.where(
        genes["tpm"] > 0, genes["weighted_len"] / genes["tpm"].where(genes["tpm"] > 0, 1), genes["mean_len"]
    )
    total_mapped = frame["NumReads"].sum()
    genes["fpkm"] = 1e9 * genes["reads"] / (genes["effective_length"] * total_mapped)
    genes.index.name = "gene_id"
    return genes[["reads", "tpm", "effective_length", "fpkm"]]


def gene_symbol_table(tx2gene: pd.DataFrame) -> pd.Series:
    """Unversioned Ensembl gene ID -> GENCODE gene_name (one name per gene)."""
    return tx2gene.drop_duplicates("gene_id").set_index("gene_id")["gene_name"].sort_index()


def collapse_to_symbols(values: pd.Series, symbols: pd.Series) -> tuple[pd.Series, int]:
    """Index gene-level values by symbol.

    Where several Ensembl genes share a symbol, keep the lowest Ensembl ID. The
    rule depends only on the annotation, never on expression, so treated and
    control profiles always use the same gene (PERD's ENCODE matrix likewise kept
    the first occurrence). Returns (values by symbol, number of dropped duplicate genes).
    """
    named = pd.DataFrame({"symbol": symbols.reindex(values.index), "value": values}).dropna(subset=["symbol"])
    named = named.sort_index()
    duplicated = named["symbol"].duplicated(keep="first")
    kept = named.loc[~duplicated].set_index("symbol")["value"]
    kept.index.name = "gene_symbol"
    return kept, int(duplicated.sum())


def log2_fpkm(fpkm: pd.Series | pd.DataFrame):
    return np.log2(fpkm + 1)
