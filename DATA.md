# Data

Everything here is public. The full raw data are about 100 GB, so they are not in the repository; this page says
where each piece comes from and where the scripts expect it. The small toy dataset in `submission/toy_data/` is enough to
run the whole method (see the README).

All coordinates are GRCh38 (hg38). Every downloaded file is checked against the md5 or sha256 recorded in the
manifests under `data/`.

## 1. PERD training tables (167 untreated ENCODE samples)

These come from my Python reproduction of PERD (a separate repository, `perd`). The scripts here read them from
`../perd/data/interim/`, i.e. the reproduction is cloned next to this repository. The tables and their sha256 are
listed in `data/baseline_manifest.json`.

| File | Contents | Built from |
|---|---|---|
| `rna_167_cells_symbol.parquet` | log2(FPKM + 1), genes × 167 samples | BIRD v3.0 ENCODE RNA-seq |
| `enhancer_openness.parquet` | openness, enhancers × 167 samples (highest 200 bp bin fully inside the enhancer) | BIRD v3.0 ENCODE DNase-seq, lifted from hg19 to hg38 |
| `genehancer_enhancers.parquet` | enhancer coordinates and target genes | GeneHancer v4.4 |
| `enhancer_tf_binding.parquet` | TFs with a ChIP-seq peak inside each enhancer | UCSC ENCODE TF clusters, lifted to hg38 |
| `netopen_training_enhancers.parquet` | the 120,100 enhancers and their input genes | the tables above |
| `netopen_rf_predictions.parquet` | PERD-RF predictions on the paper's 57 random test samples | the tables above |

Original sources:

| Source | Link |
|---|---|
| BIRD data v3.0 (ENCODE RNA-seq and DNase-seq, 167 samples) | https://github.com/WeiqiangZhou/BIRD-data/releases/download/v3.0/BIRD_data_ENCODE.zip |
| GeneHancer v4.4 (as distributed with PERD) | https://github.com/wangyc82/PERD |
| ENCODE TF ChIP-seq clusters (UCSC, hg19) | http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeRegTfbsClustered/wgEncodeRegTfbsClusteredV3.bed.gz |
| hg19 → hg38 liftover chain | https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz |

## 2. ENCODE perturbation series (the test data)

Downloaded from the ENCODE portal (https://www.encodeproject.org) by the download scripts. Each manifest lists every
file with its experiment, replicate, md5 and URL (`https://www.encodeproject.org/files/<accession>/@@download/...`).

| Series | Manifest | Files | Size | File types |
|---|---|---|---|---|
| A549 + dexamethasone, 0–12 h | `data/encode_dex_manifest.json` | 97 | 29.4 GB | DNase read-depth normalized signal (bigWig), RNA gene quantifications |
| K562 + drugs, DMSO controls | `data/encode_k562_manifest.json` | 196 | 31.7 GB | ATAC alignments (BAM), RNA gene quantifications |
| T-cell activation (5 T-cell types) | `data/encode_tcell_manifest.json` | 34 | 7.4 GB | DNase signal, RNA gene quantifications |
| HCT116 auxin degrons | `data/encode_hct116_manifest.json` | 58 | 18.0 GB | DNase signal, RNA gene quantifications |
| SK-N-SH + retinoic acid | `data/encode_sknsh_manifest.json` | 13 | 2.0 GB | DNase signal, RNA gene quantifications |
| A549 GR ChIP-seq peaks (11 time points) | `data/encode_gr_manifest.json` | 11 | small | conservative IDR thresholded peaks (bed) |

The large signal and BAM files are reduced to per-bin tables during processing and then deleted, so the processed
data under `data/processed/` take only a few GB.

A549 dexamethasone experiments (DNase / RNA): 0 h ENCSR136DNA / ENCSR632DQP; 0.5 h ENCSR406EMB / ENCSR070IYV;
1 h ENCSR384KCZ / ENCSR656FIH; 2 h ENCSR837VHE / ENCSR924BHF; 3 h ENCSR294XUZ / ENCSR624RID; 4 h ENCSR599WJC /
ENCSR326PTW; 5 h ENCSR565WPR / ENCSR543QRU; 6 h ENCSR077EYC / ENCSR255VBV; 7 h ENCSR347CEH / ENCSR385TFN; 8 h
ENCSR660OQE / ENCSR546PPG; 10 h ENCSR128IVG / ENCSR224PTG; 12 h ENCSR523FJT / ENCSR154TDP. The other series'
experiment accessions are in their manifests.

## 3. TF network and motifs

| Source | Link | Notes |
|---|---|---|
| CollecTRI TF–target network (via OmniPath) | https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&loops=yes&format=tsv | retrieved 2026-10-05, 64,979 rows, md5 in `data/collectri_manifest.json`; saved as `data/raw/collectri/collectri.tsv` |
| Motif archetypes v1.0 (Vierstra et al., 2020) | https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/ | `hg38.archetype_motifs.v1.0.bed.gz` (12 GB) with its `.tbi` index, and `motif_annotations.xlsx`; see the commands below |

Motif download (about 12 GB):

```bash
mkdir -p data/raw/motifs && cd data/raw/motifs
B=https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0
for f in motif_annotations.xlsx hg38.archetype_motifs.v1.0.bed.gz.tbi hg38.archetype_motifs.v1.0.bed.gz; do
  curl -sS --retry 5 -O $B/$f
done
```

## 4. Toy data

`submission/toy_data/` holds a few MB cut from the real processed data by `submission/make_toy_data.py`: 150 random enhancers
from PERD's candidate set with their openness in all 167 ENCODE samples, the RNA of the genes they and the TF network
need, three perturbation series (A549 dexamethasone at 2, 6 and 12 h; T-cell activation; three HCT116 degrons), the
full CollecTRI network, and the motif tables for those enhancers. See `submission/toy_data/README.md`.
