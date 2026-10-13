"""Cut a small toy dataset out of the full processed data, for submission/run_demo.py.

The toy set keeps the real structure but is small enough for git (a few MB):
  - 150 enhancers drawn at random (seed 42) from PERD's candidate set F1, with their ENCODE openness in all 167 cells;
  - the ENCODE RNA of every gene those enhancers use, plus every CollecTRI target gene (so TF activity can be scored);
  - three perturbation series: A549 + dexamethasone (0 h vs 2, 6, 12 h), T-cell activation (5 cell types),
    HCT116 degrons (BRD4, CTCF, MED14), as replicate-level openness and RNA tables;
  - the full CollecTRI network and the motif-archetype tables for the chosen enhancers.

This needs the full data (see DATA.md) and is only for maintainers; the toy files it writes are committed, so users
never run it. Usage: pixi run make-toy-data  [--perd-dir ../perd/data/interim]
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"  # set by --data-root
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "submission" / "toy_data"
N_ENHANCERS, SEED, MIN_KB = 150, 42, 0.2
A549_TIMES = (2.0, 6.0, 12.0)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def conditions() -> list[dict]:
    """Toy perturbation conditions with the sample (file accession) names of each side."""
    out = []
    files = pd.DataFrame(json.loads((ROOT / "data" / "encode_dex_manifest.json").read_text())["files"])
    pick = lambda assay, t: files[(files["assay"] == assay) & (files["time_h"] == t)]["file_accession"].tolist()  # noqa: E731
    for t in A549_TIMES:
        out.append({"dataset": "encode_dex", "unit": "A549 dexamethasone", "name": f"A549 dex {t:g}h",
                    "acc_control": pick("DNase", 0.0), "acc_treated": pick("DNase", t), "rna_control": pick("RNA", 0.0), "rna_treated": pick("RNA", t)})
    s = pd.read_csv(PROC / "encode_tcell" / "samples.csv")
    for pair, g in s.groupby("pair"):
        a, r = g[g["assay"] == "DNase-seq"], g[g["assay"] != "DNase-seq"]
        sel = lambda d, st: d[d["state"] == st]["file_accession"].tolist()  # noqa: E731
        out.append({"dataset": "encode_tcell", "unit": "T-cell activation", "name": f"T {pair}",
                    "acc_control": sel(a, "resting"), "acc_treated": sel(a, "activated"), "rna_control": sel(r, "resting"), "rna_treated": sel(r, "activated")})
    s = pd.read_csv(PROC / "encode_hct116" / "samples.csv")
    gate = pd.read_csv(TABLES / "a09_gate_per_unit.csv")
    for unit in gate[(gate["dataset"] == "hct116") & gate["replicable"]]["unit"]:
        g = s[s["unit"] == unit]
        a, r = g[g["assay"] == "DNase-seq"], g[g["assay"] != "DNase-seq"]
        sel = lambda d, st: d[d["state"] == st]["file_accession"].tolist()  # noqa: E731
        out.append({"dataset": "encode_hct116", "unit": f"HCT116 {unit}", "name": f"HCT116 {unit}",
                    "acc_control": sel(a, "untreated"), "acc_treated": sel(a, "treated"), "rna_control": sel(r, "untreated"), "rna_treated": sel(r, "treated")})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perd-dir", type=Path, default=ROOT.parent / "perd" / "data" / "interim")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="folder holding processed/ and raw/ (default ./data)")
    args = parser.parse_args()
    global PROC, RAW
    PROC, RAW = args.data_root / "processed", args.data_root / "raw"
    perd = args.perd_dir
    OUT.mkdir(parents=True, exist_ok=True)

    # Enhancers: random F1 candidates (the A549 candidate set, as in the real analysis).
    pipeline = load_module("pipeline", ROOT / "scripts" / "11b_perd_pipeline_dex.py")
    preds = pd.read_parquet(PROC / "encode_dex" / "perd_predictions_rf.parquet", columns=["skip_reason"])
    fitted = preds.index[preds["skip_reason"] == ""]
    symbols = set(pd.read_parquet(PROC / "encode_dex" / "rna_log2fpkm_replicates.parquet").index)
    f1 = pd.Index(sorted(pipeline.candidate_sets(fitted, symbols)[0]["F1"]))
    enh = pd.Index(sorted(np.random.default_rng(SEED).choice(f1, N_ENHANCERS, replace=False)))

    training = pd.read_parquet(perd / "netopen_training_enhancers.parquet").set_index("enhancer_id").loc[enh]
    gh = pd.read_parquet(perd / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id").loc[enh]
    binding = pd.read_parquet(perd / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"].reindex(enh)
    as_list = lambda v: [] if v is None or (isinstance(v, float) and np.isnan(v)) else [str(x) for x in v]  # noqa: E731
    enhancers = pd.DataFrame({"enhancer_id": enh, "chrom": gh["chrom"].to_numpy(), "start": gh["start"].to_numpy(), "end": gh["end"].to_numpy(),
                              "usable_genes": [as_list(v) for v in training["usable_genes"]],
                              "target_genes": [as_list(v) for v in gh["target_genes"]], "tf_genes": [as_list(v) for v in binding]})
    enhancers.to_parquet(OUT / "enhancers.parquet", index=False)

    # ENCODE training tables. Genes: everything the enhancers use plus every CollecTRI target.
    raw_net = pd.read_csv(RAW / "collectri" / "collectri.tsv", sep="\t")
    encode_rna = pd.read_parquet(perd / "rna_167_cells_symbol.parquet")
    used = {g for col in ("usable_genes", "target_genes", "tf_genes") for v in enhancers[col] for g in v}
    genes = encode_rna.index[encode_rna.index.isin(used | set(raw_net["target_genesymbol"].astype(str)) | set(raw_net["source_genesymbol"].astype(str)))]
    encode_rna.loc[genes].astype("float32").to_parquet(OUT / "encode_rna.parquet")
    pd.read_parquet(perd / "enhancer_openness.parquet").loc[enh].astype("float32").to_parquet(OUT / "encode_openness.parquet")
    with open(RAW / "collectri" / "collectri.tsv", "rb") as src, gzip.open(OUT / "collectri.tsv.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)

    # Motif archetypes: raw match counts for the toy enhancers, and the density z-score computed over all training
    # enhancers exactly as the real scripts do (so motif sets match the full analysis).
    all_enh = pd.Index(pd.read_parquet(perd / "netopen_training_enhancers.parquet")["enhancer_id"])
    gh_all = pd.read_parquet(perd / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")
    length = ((gh_all["end"] - gh_all["start"]) / 1000).clip(lower=MIN_KB)
    counts = pd.read_parquet(PROC / "motifs" / "enhancer_archetypes.parquet")
    counts = counts[counts["enhancer_id"].isin(all_enh)].copy()
    counts["density"] = counts["n_matches"] / counts["enhancer_id"].map(length).to_numpy()
    n = len(all_enh)
    stats = counts.groupby("archetype")["density"].agg(s1="sum", s2=lambda d: (d ** 2).sum())
    mean = stats["s1"] / n
    sd = np.sqrt((stats["s2"] / n - mean ** 2).clip(lower=1e-12))
    toy = counts[counts["enhancer_id"].isin(enh)].copy()
    toy["density_z"] = (toy["density"] - toy["archetype"].map(mean)) / toy["archetype"].map(sd)
    toy[["enhancer_id", "archetype", "n_matches", "density", "density_z"]].to_parquet(OUT / "motif_counts.parquet", index=False)
    pd.DataFrame({"archetype": mean.index, "density_mean": mean.to_numpy(), "density_sd": sd.to_numpy()}).to_csv(OUT / "motif_density_scale.csv", index=False)
    shutil.copy(TABLES / "motif_archetype_tfs.csv", OUT / "motif_archetype_tfs.csv")

    # Perturbation tables (replicate level), restricted to the toy enhancers and genes.
    conds = conditions()
    opens, rnas = [], []
    for dataset in sorted({c["dataset"] for c in conds}):
        acc = sorted({a for c in conds if c["dataset"] == dataset for a in c["acc_control"] + c["acc_treated"]})
        rs = sorted({a for c in conds if c["dataset"] == dataset for a in c["rna_control"] + c["rna_treated"]})
        opens.append(pd.read_parquet(PROC / dataset / "enhancer_openness_replicates.parquet").reindex(enh)[acc])
        rnas.append(pd.read_parquet(PROC / dataset / "rna_log2fpkm_replicates.parquet").reindex(genes)[rs])
    pd.concat(opens, axis=1).astype("float32").to_parquet(OUT / "perturbation_openness.parquet")
    pd.concat(rnas, axis=1).astype("float32").to_parquet(OUT / "perturbation_rna.parquet")
    (OUT / "conditions.json").write_text(json.dumps(conds, indent=1) + "\n")

    sizes = {p.name: p.stat().st_size for p in sorted(OUT.iterdir())}
    print(json.dumps({"enhancers": len(enh), "genes": len(genes), "conditions": len(conds), "bytes": sizes, "total_mb": round(sum(sizes.values()) / 1e6, 2)}, indent=1))


if __name__ == "__main__":
    main()
