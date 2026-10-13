"""Amendment 03 sections B and C: build RNA and DNase replicate tables, then capability checks (no PERD output).

Definitions were fixed in execution.md (2026-10-04) before this script first ran.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from perd_perturbation import atac, encode_dex, evaluation, measures, rna


ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
MANIFEST = ROOT / "data" / "encode_dex_manifest.json"
RAW = ROOT / "data" / "raw" / "encode_dex"
OUT = ROOT / "data" / "processed" / "encode_dex"
TABLES = ROOT / "results" / "tables"
REFERENCE = ROOT / "data" / "raw" / "reference"
MIN_MEAN = 1.0
MIN_RELIABILITY = 0.5
POSITIVE_CONTROL = ["FKBP5", "PER1", "TSC22D3", "DUSP1", "ZBTB16"]
POSITIVE_WINDOW = [1, 2, 3, 4, 5, 6, 7, 8]


def condition_mean(table: pd.DataFrame, accessions: list[str]) -> pd.Series:
    return table[accessions].mean(axis=1)


def reliability(openness: pd.DataFrame, control: list[tuple[int, str]], treated: list[tuple[int, str]]) -> dict:
    """Replicate-half reliability of the measured change (C1); replicates as (number, accession)."""
    full = (condition_mean(openness, [a for _, a in control]) + condition_mean(openness, [a for _, a in treated])) / 2
    evaluable = full.notna() & (full >= MIN_MEAN)
    ctrl_a, ctrl_b = encode_dex.replicate_halves([n for n, _ in control])
    trt_a, trt_b = encode_dex.replicate_halves([n for n, _ in treated])
    by_number_c, by_number_t = dict(control), dict(treated)
    delta_a = condition_mean(openness, [by_number_t[n] for n in trt_a]) - condition_mean(openness, [by_number_c[n] for n in ctrl_a])
    delta_b = condition_mean(openness, [by_number_t[n] for n in trt_b]) - condition_mean(openness, [by_number_c[n] for n in ctrl_b])
    r = evaluation.correlation(delta_a[evaluable], delta_b[evaluable])
    rel = measures.spearman_brown(r)
    return {
        "evaluable": int(evaluable.sum()),
        "halves_treated": [trt_a, trt_b],
        "halves_control": [ctrl_a, ctrl_b],
        "split_half_spearman": r,
        "reliability": rel,
        "ceiling_sqrt_reliability": float(np.sqrt(rel)) if rel > 0 else 0.0,
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    files = pd.DataFrame(manifest["files"])
    status_dir = ROOT / "data" / "interim" / "encode_dex_status"
    statuses = {f.stem: json.loads(f.read_text())["status"] for f in status_dir.glob("*.json")}
    if len(statuses) != len(files) or any(s != "done" for s in statuses.values()):
        raise SystemExit(f"Download incomplete: {pd.Series(statuses).value_counts().to_dict()}")
    times = sorted(files["time_h"].unique())
    out: dict = {"times_h": [float(t) for t in times]}

    # ---- B1 RNA ----
    tx2gene = rna.read_gencode_tx2gene(REFERENCE / "gencode.v44.transcripts.fa.gz")
    symbols = rna.gene_symbol_table(tx2gene)
    rna_files = files[files["assay"] == "RNA"].sort_values(["time_h", "biological_replicate"])
    rna_columns, dropped = {}, {}
    for row in rna_files.itertuples():
        rna_columns[row.file_accession], dropped[row.file_accession] = encode_dex.encode_log2fpkm_by_symbol(RAW / "rna" / f"{row.file_accession}.tsv", symbols)
    rna_table = pd.DataFrame(rna_columns).fillna(0.0)
    rna_table.to_parquet(OUT / "rna_log2fpkm_replicates.parquet")
    usable = sorted({g for genes in pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["usable_genes"] for g in genes})
    found = sum(g in rna_table.index for g in usable)
    out["B1_rna"] = {"replicates": int(rna_table.shape[1]), "genes": int(rna_table.shape[0]), "usable_gene_coverage": found / len(usable), "usable_genes_found": found, "usable_genes_total": len(usable), "duplicate_symbol_genes_dropped": int(np.median(list(dropped.values())))}

    # ---- B2/B3 DNase: enhancer raw max, scale constant, openness per replicate ----
    grid = pd.read_parquet(OUT / "grid.parquet")
    genehancer = pd.read_parquet(PERD / "genehancer_enhancers.parquet").set_index("enhancer_id")
    training_ids = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"]
    training = genehancer.loc[training_ids].reset_index()
    dnase_files = files[files["assay"] == "DNase"].sort_values(["time_h", "biological_replicate"])
    per_bin = {acc: grid.assign(value=np.load(OUT / "bins" / f"{acc}.npy").astype(float)) for acc in dnase_files["file_accession"]}
    raw = atac.enhancer_openness(training, per_bin)
    raw.to_parquet(OUT / "enhancer_raw_max_replicates.parquet")
    control_acc = dnase_files.loc[dnase_files["time_h"] == 0, "file_accession"].tolist()
    perd_openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    perd_values = perd_openness.to_numpy().ravel()
    perd_median_positive = float(np.median(perd_values[perd_values > 0]))
    control_raw = raw[control_acc].mean(axis=1)
    control_raw_positive_median = float(control_raw[control_raw > 0].median())
    k = encode_dex.scale_constant(perd_median_positive, control_raw_positive_median)
    out["B3_scale"] = {"perd_median_positive": perd_median_positive, "control_raw_positive_median": control_raw_positive_median, "k": k}

    def openness_at(scale: float) -> pd.DataFrame:
        return np.log2(scale * raw + 1)

    openness = openness_at(k)
    openness.to_parquet(OUT / "enhancer_openness_replicates.parquet")

    replicates = {t: list(zip(g["biological_replicate"], g["file_accession"])) for t, g in dnase_files.groupby("time_h")}
    control = replicates[0]

    # ---- C1/C2 reliability per time point (k, with k/2 and 2k sensitivity) ----
    rows = []
    for t in times[1:]:
        for label, scale in (("k", k), ("k/2", k / 2), ("2k", 2 * k)):
            rows.append({"time_h": t, "scale": label, "n_treated_replicates": len(replicates[t]), **reliability(openness_at(scale), control, replicates[t])})
    rel = pd.DataFrame(rows)
    rel.to_csv(TABLES / "encode_dex_reliability.csv", index=False)
    primary_rows = rel[(rel["scale"] == "k") & (rel["reliability"] >= MIN_RELIABILITY)]
    primary = [float(t) for t in primary_rows["time_h"]]
    out["C1_reliability"] = rel[rel["scale"] == "k"].drop(columns="scale").to_dict("records")
    out["C2_primary_time_points"] = primary
    out["C2_verdict"] = "at least one primary time point" if primary else "the data are insufficient for enhancer-level validation"

    # ---- C3 control-half noise ----
    ctrl_a, ctrl_b = encode_dex.replicate_halves([n for n, _ in control])
    by_number = dict(control)
    half_a, half_b = condition_mean(openness, [by_number[n] for n in ctrl_a]), condition_mean(openness, [by_number[n] for n in ctrl_b])
    control_mean = condition_mean(openness, [a for _, a in control])
    ok = control_mean.notna() & (control_mean >= MIN_MEAN)
    diff = (half_a - half_b)[ok].abs()
    out["C3_control_noise"] = {"enhancers": int(ok.sum()), "abs_median": float(diff.median()), "abs_p95": float(diff.quantile(0.95)), "event_threshold_p99": float(diff.quantile(0.99))}

    # ---- C4 promoter control (each time point computed; primary ones flagged) ----
    promoters = pd.read_parquet(OUT / "promoters.parquet")
    promoter_raw = pd.DataFrame({acc: np.load(OUT / "promoters" / f"{acc}.npy").astype(float) for acc in dnase_files["file_accession"]})
    promoter_open = np.log2(k * promoter_raw + 1)
    rna_reps = {t: g["file_accession"].tolist() for t, g in rna_files.groupby("time_h")}
    fpkm = 2**rna_table - 1
    gene_names = promoters["gene_name"].to_numpy()
    in_rna = np.isin(gene_names, rna_table.index)
    c4 = []
    for t in times[1:]:
        p_ctrl = promoter_open[[a for _, a in control]].mean(axis=1).to_numpy()
        p_trt = promoter_open[[a for _, a in replicates[t]]].mean(axis=1).to_numpy()
        names = gene_names[in_rna]
        r_ctrl = rna_table.loc[names, rna_reps[0]].mean(axis=1).to_numpy()
        r_trt = rna_table.loc[names, rna_reps[t]].mean(axis=1).to_numpy()
        f_ctrl = fpkm.loc[names, rna_reps[0]].mean(axis=1).to_numpy()
        f_trt = fpkm.loc[names, rna_reps[t]].mean(axis=1).to_numpy()
        prom_change = (p_trt - p_ctrl)[in_rna]
        prom_mean = ((p_trt + p_ctrl) / 2)[in_rna]
        keep = (np.maximum(f_ctrl, f_trt) >= 1) & (prom_mean >= MIN_MEAN)
        result = measures.spearman_bootstrap(prom_change[keep], (r_trt - r_ctrl)[keep])
        c4.append({"time_h": t, "primary": t in primary, **result})
    out["C4_promoter_control"] = c4

    # ---- C5 RNA positive control ----
    c5_rows = []
    for gene in POSITIVE_CONTROL:
        for t in times[1:]:
            if gene not in rna_table.index:
                c5_rows.append({"gene": gene, "time_h": t, "log2fc": np.nan, "p": np.nan, "induced": False, "missing": True})
                continue
            treated_values, control_values = rna_table.loc[gene, rna_reps[t]].to_numpy(), rna_table.loc[gene, rna_reps[0]].to_numpy()
            p = float(ttest_ind(treated_values, control_values, equal_var=False).pvalue)
            fc = float(treated_values.mean() - control_values.mean())
            c5_rows.append({"gene": gene, "time_h": t, "log2fc": fc, "p": p, "induced": bool(fc > 0 and p < 0.05), "missing": False})
    c5 = pd.DataFrame(c5_rows)
    c5.to_csv(TABLES / "encode_dex_rna_positive_control.csv", index=False)
    passed_genes = sorted(c5[(c5["time_h"].isin(POSITIVE_WINDOW)) & c5["induced"]]["gene"].unique())
    out["C5_rna_positive_control"] = {"genes_induced_between_1_and_8h": passed_genes, "n_passed": len(passed_genes), "control_passes": len(passed_genes) >= 3, "max_log2fc_by_gene": c5.groupby("gene")["log2fc"].max().round(3).to_dict()}

    # ---- C6 depth and QC ----
    depth_rows = []
    for row in files.sort_values(["assay", "time_h", "biological_replicate"]).itertuples():
        metrics = manifest["alignment_depth"].get(row.experiment, {}).get(str(row.biological_replicate), {})
        depth_rows.append({"assay": row.assay, "time_h": row.time_h, "biological_replicate": row.biological_replicate, "file_accession": row.file_accession, **{k: v for k, v in metrics.items() if k != "alignment_file"}})
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(TABLES / "encode_dex_depth.csv", index=False)
    evaluable_fraction = {float(t): r["evaluable"] / len(training_ids) for t, r in zip(rel.loc[rel["scale"] == "k", "time_h"], out["C1_reliability"])}
    qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    perd_training = perd_openness.loc[training_ids].to_numpy().ravel()
    control_open = condition_mean(openness, [a for _, a in control]).dropna()
    out["C6_qc"] = {
        "evaluable_fraction_of_training_enhancers": evaluable_fraction,
        "control_openness_quantiles": dict(zip(map(str, qs), np.round(np.quantile(control_open, qs), 3))),
        "perd_training_openness_quantiles": dict(zip(map(str, qs), np.round(np.quantile(perd_training, qs), 3))),
        "control_fraction_zero": float((control_open == 0).mean()),
        "perd_fraction_zero": float((perd_training == 0).mean()),
        "control_enhancers_defined": int(control_open.shape[0]),
        "depth_reported_replicates": int(depth.drop(columns=["assay", "time_h", "biological_replicate", "file_accession"]).notna().any(axis=1).sum()),
    }
    print(json.dumps(out, indent=2, default=float))
    (TABLES / "encode_dex_capability.json").write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
