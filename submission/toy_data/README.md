# Toy data

A small slice of the real data, made by `submission/make_toy_data.py` (seed 42) and used by `submission/demo.ipynb` and `submission/run_demo.py`. The values
are real, not simulated; only the size is reduced.

| File | Contents |
|---|---|
| `enhancers.parquet` | 150 enhancers drawn at random from PERD's candidate set: coordinates (hg38), PERD input genes, GeneHancer target genes, ChIP-bound TFs |
| `encode_openness.parquet` | openness of those enhancers in the 167 ENCODE training samples |
| `encode_rna.parquet` | log2(FPKM + 1) in the 167 samples for the genes the enhancers use plus every CollecTRI gene |
| `collectri.tsv.gz` | the full CollecTRI TF–target network (OmniPath, 2026-10-05) |
| `motif_counts.parquet` | motif-archetype matches per enhancer, with density per kb and its z-score over all 120,100 training enhancers |
| `motif_density_scale.csv` | mean and SD of each archetype's density over the 120,100 enhancers (gives the z-score of an absent motif) |
| `motif_archetype_tfs.csv` | archetype → TFs (Vierstra et al., 2020) |
| `perturbation_openness.parquet` | replicate-level openness of the 150 enhancers in the toy perturbation samples |
| `perturbation_rna.parquet` | replicate-level log2(FPKM + 1) for the same samples |
| `conditions.json` | 11 before/after pairs in 5 perturbations: A549 + dexamethasone (2, 6, 12 h), T-cell activation (5 cell types), HCT116 BRD4 / CTCF / MED14 degrons; each lists its control and treated sample names |

What differs from the full analysis: 150 instead of 120,100 enhancers, 5 instead of 16 perturbations, and TF
activity is scored on the genes kept here rather than on all ~28,000, so activity values differ slightly from the
full run. The demo's numbers will therefore not match the report.
