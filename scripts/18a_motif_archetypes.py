"""Motif features for amendment 10 (definitions in execution.md, 2026-10-06; PERD-blind, no openness read).

Vierstra et al. 2020 motif archetypes v1.0 (hg38): per training enhancer x archetype, the number of matches whose
midpoint lies inside the enhancer (bedtools -f 0.5) and the maximum match score. Archetype -> TF symbols from the
annotation table. Sanity: GR-family archetype enrichment at A549 GR ChIP-seq enhancers; TF coverage gained.
Run in the `bio` pixi environment (tabix, bedtools).
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
RAW = ROOT / "data" / "raw" / "motifs"
OUT = ROOT / "data" / "processed" / "motifs"
TABLES = ROOT / "results" / "tables"
BED = RAW / "hg38.archetype_motifs.v1.0.bed.gz"


def motif_symbols(name: str) -> list[str]:
    """TF gene symbols from a motif name, e.g. 'LHX2_MOUSE.H11MO.0.A' -> ['LHX2'], 'Ar::Nr3c1_MA..' -> ['AR', 'NR3C1']."""
    head = re.split(r"[_.]", name, maxsplit=1)[0]
    return [s.upper() for s in head.split("::") if s]


def archetype_tfs() -> pd.DataFrame:
    sheets = pd.read_excel(RAW / "motif_annotations.xlsx", sheet_name=None)
    clusters, motifs = sheets["Archetype clusters"], sheets["Motifs"]
    rows = []
    for cid, group in motifs.groupby("Cluster_ID"):
        tfs = sorted({s for m in group["Motif"] for s in motif_symbols(str(m))})
        name = clusters.loc[clusters["Cluster_ID"] == cid, "Name"].iloc[0]
        dbd = clusters.loc[clusters["Cluster_ID"] == cid, "DBD"].iloc[0]
        rows.append({"cluster_id": int(cid), "archetype": name, "dbd": dbd, "n_motifs": len(group), "tfs": ";".join(tfs)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"]
    gh = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id").loc[training]
    enh = gh[["chrom", "start", "end"]].reset_index()[["chrom", "start", "end", "enhancer_id"]].sort_values(["chrom", "start"])
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        enh_bed, regions = Path(tmp) / "enhancers.bed", Path(tmp) / "regions.bed"
        enh.to_csv(enh_bed, sep="\t", header=False, index=False)
        subprocess.run(f"bedtools merge -i {enh_bed} > {regions}", shell=True, check=True)
        aggregated = OUT / "enhancer_archetype_counts.tsv"
        # tabix pulls only records overlapping enhancers; bedtools keeps matches with >= 50% inside an enhancer;
        # awk aggregates count and max score per (enhancer, archetype).
        cmd = (f"tabix {BED} -R {regions} | bedtools intersect -a stdin -b {enh_bed} -wa -wb -f 0.5 | "
               "awk -F'\\t' 'BEGIN{OFS=\"\\t\"} {k=$12\"\\t\"$4; n[k]++; if(!(k in m) || $5>m[k]) m[k]=$5} END{for(k in n) print k, n[k], m[k]}' "
               f"> {aggregated}")
        subprocess.run(["bash", "-c", "set -euo pipefail; " + cmd], check=True)
    counts = pd.read_csv(aggregated, sep="\t", header=None, names=["enhancer_id", "archetype", "n_matches", "max_score"])
    counts.to_parquet(OUT / "enhancer_archetypes.parquet", index=False)
    aggregated.unlink()
    mapping = archetype_tfs()
    mapping.to_csv(TABLES / "motif_archetype_tfs.csv", index=False)

    # Coverage: activity TFs that gain enhancers through motifs, vs through the ChIP binding list.
    activity_tfs = set(pd.read_parquet(ROOT / "data" / "processed" / "tf_activity" / "activity_encode167.parquet").index)
    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet")
    chip_tfs = {t for v in binding["tf_genes"] for t in v}
    present = set(counts["archetype"])
    motif_tfs = {tf for _, r in mapping.iterrows() if r["archetype"] in present for tf in str(r["tfs"]).split(";") if tf}
    coverage = {"enhancers_with_any_match": int(counts["enhancer_id"].nunique()), "archetypes_present": len(present), "pairs": int(len(counts)),
                "median_archetypes_per_enhancer": float(counts.groupby("enhancer_id").size().median()),
                "activity_tfs": len(activity_tfs), "activity_tfs_with_chip_binding": len(activity_tfs & chip_tfs),
                "activity_tfs_with_motif": len(activity_tfs & motif_tfs), "activity_tfs_with_either": len(activity_tfs & (chip_tfs | motif_tfs)),
                "examples": {tf: sorted(mapping[mapping["tfs"].str.split(";").map(lambda s, t=tf: t in s)]["archetype"].tolist()) for tf in ("NR3C1", "RARA", "TP53", "AR", "NFKB1", "SOX2", "KLF4")}}

    # Sanity: GR-family archetype(s) vs A549 GR ChIP-seq (union of 11 experiments).
    spec = importlib.util.spec_from_file_location("gr", ROOT / "scripts" / "11c_dex_subsets_modules_gr.py")
    gr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gr)
    ids = pd.Index(training)
    gr_union, _ = gr.gr_masks(ids)
    gr_arch = [a for a in coverage["examples"]["NR3C1"]]
    has_motif = ids.isin(counts.loc[counts["archetype"].isin(gr_arch), "enhancer_id"])
    a, b = int((has_motif & gr_union).sum()), int((has_motif & ~gr_union).sum())
    c, d = int((~has_motif & gr_union).sum()), int((~has_motif & ~gr_union).sum())
    coverage["gr_sanity"] = {"gr_archetypes": gr_arch, "frac_with_motif_given_gr_peak": a / (a + c), "frac_with_motif_no_gr_peak": b / (b + d), "odds_ratio": (a * d) / max(b * c, 1)}
    (TABLES / "motif_coverage.json").write_text(json.dumps(coverage, indent=2, default=float) + "\n")
    print(json.dumps(coverage, indent=1, default=float))
    print(counts.groupby("archetype").size().sort_values(ascending=False).head(15).to_string())
    print(np.round(counts[["n_matches", "max_score"]].describe(), 2).to_string())


if __name__ == "__main__":
    main()
