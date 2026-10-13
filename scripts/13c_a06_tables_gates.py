"""Amendment 06 B-C: replicate tables, PERD-blind gates, control noise, and RNA positive controls (no PERD output).

K562: ATAC enhancer openness per replicate (BIRD normalization, max over contained bins); RNA log2(FPKM+1) per
replicate; per-condition replicate-split extreme test (gate); DMSO rep1-vs-rep2 noise per time; CDKN1A under HDAC
inhibitors. T cells: DNase enhancer openness per replicate (k from pooled resting conditions); RNA per replicate;
cross-pair extreme-test gate; resting replicate noise; IL2RA/IRF4/MYC/CD69 induction.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from perd_perturbation import atac, encode_dex, plate_eval, rna

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
TABLES = ROOT / "results" / "tables"
REFERENCE = RAW / "reference"
MIN_MEAN = 1.0
HDAC = ["Panobinostat", "Vorinostat", "AR-42"]
TCELL_MARKERS = ["IL2RA", "IRF4", "MYC", "CD69"]


def training_enhancers() -> pd.DataFrame:
    genehancer = pd.read_parquet(PERD / "genehancer_enhancers.parquet").set_index("enhancer_id")
    ids = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"]
    return genehancer.loc[ids].reset_index()


def rna_table(records: list[dict], dataset: str, symbols: pd.Series) -> pd.DataFrame:
    columns = {}
    for r in records:
        columns[r["file_accession"]], _ = encode_dex.encode_log2fpkm_by_symbol(RAW / f"encode_{dataset}" / "rna" / f"{r['file_accession']}.tsv", symbols)
    return pd.DataFrame(columns).fillna(0.0)


def gate_test(key, ranking, measured, stratum):
    ks = plate_eval.k_values(len(ranking))
    result = plate_eval.extreme_test_components(ranking, measured, stratum, {0.01: ks[0.01]}, n_perm=2000, seed=42)[0.01]
    return key, result


def deciles(values: np.ndarray) -> np.ndarray:
    return pd.qcut(pd.Series(values).rank(method="first").to_numpy(), 10, labels=False)


def rep_number(r: dict) -> int:
    return int(str(r["biological_replicate"]).split(",")[0])


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    TABLES.mkdir(parents=True, exist_ok=True)
    statuses = [json.loads(p.read_text()) for p in (ROOT / "data" / "interim" / "a06_status").glob("*.json")]
    bad = [s["file_accession"] for s in statuses if s["status"] != "done"]
    if bad:
        raise SystemExit(f"Download incomplete or failed: {bad[:10]}")
    training = training_enhancers()
    tx2gene = rna.read_gencode_tx2gene(REFERENCE / "gencode.v44.transcripts.fa.gz")
    symbols = rna.gene_symbol_table(tx2gene)
    usable = sorted({g for genes in pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["usable_genes"] for g in genes})
    report: dict = {}
    tasks = []

    # ================= K562 =================
    k562 = json.loads((ROOT / "data" / "encode_k562_manifest.json").read_text())["files"]
    out = PROCESSED / "encode_k562"
    k_rna = [r for r in k562 if r["assay"] == "total RNA-seq"]
    k_atac = [r for r in k562 if r["assay"] == "ATAC-seq"]
    rna_k = rna_table(k_rna, "k562", symbols)
    rna_k.to_parquet(out / "rna_log2fpkm_replicates.parquet")
    per_bin = {}
    for r in k_atac:
        bins, meta = atac.load_bins(out / "atac_bins" / f"{r['file_accession']}.npz")
        per_bin[r["file_accession"]] = bins.assign(value=atac.normalize(bins["count"], meta["reads_final"]))
    openness_k = pd.concat([atac.enhancer_openness(training, dict(list(per_bin.items())[i : i + 12])) for i in range(0, len(per_bin), 12)], axis=1)
    openness_k.to_parquet(out / "enhancer_openness_replicates.parquet")
    samples_k = pd.DataFrame([{k: r[k] for k in ("file_accession", "assay", "drug", "dose", "time_h", "group", "experiment")} | {"rep": rep_number(r)} for r in k562])
    samples_k.to_csv(out / "samples.csv", index=False)
    report["k562"] = {"rna_replicates": len(k_rna), "atac_replicates": len(k_atac), "usable_gene_coverage": float(np.mean([g in rna_k.index for g in usable]))}

    atac_s = samples_k[samples_k["assay"] == "ATAC-seq"]
    noise = {}
    for t, g in atac_s[atac_s["drug"] == "DMSO"].groupby("time_h"):
        first = g[g["experiment"] == sorted(g["experiment"])[0]].sort_values("rep")
        if len(first) >= 2:
            a, b = openness_k[first["file_accession"].iloc[0]], openness_k[first["file_accession"].iloc[1]]
            ok = a.notna() & b.notna() & ((a + b) / 2 >= MIN_MEAN)
            d = (a - b)[ok].abs()
            noise[float(t)] = {"p50": float(d.median()), "p95": float(d.quantile(0.95)), "event_threshold_p99": float(d.quantile(0.99)), "n": int(ok.sum())}
    report["k562"]["dmso_noise_by_time"] = noise

    for (drug, t), g in atac_s[atac_s["drug"] != "DMSO"].groupby(["drug", "time_h"]):
        dmso = atac_s[(atac_s["drug"] == "DMSO") & (atac_s["time_h"] == t)]
        first_dmso = dmso[dmso["experiment"] == sorted(dmso["experiment"])[0]].set_index("rep")["file_accession"]
        treated = g.set_index("rep")["file_accession"]
        if not {1, 2} <= set(treated.index) or not {1, 2} <= set(first_dmso.index):
            continue
        control_mean = openness_k[dmso["file_accession"]].mean(axis=1)
        treated_mean = openness_k[treated.to_list()].mean(axis=1)
        evaluable = control_mean.notna() & treated_mean.notna() & ((control_mean + treated_mean) / 2 >= MIN_MEAN)
        delta_a = (openness_k[treated[1]] - openness_k[first_dmso[1]])[evaluable].to_numpy()
        delta_b = (openness_k[treated[2]] - openness_k[first_dmso[2]])[evaluable].to_numpy()
        tasks.append((("k562", drug, float(t)), delta_a, delta_b, deciles(control_mean[evaluable].to_numpy())))

    rna_s = samples_k[samples_k["assay"] == "total RNA-seq"]
    controls = []
    for drug in HDAC:
        for t in sorted(rna_s["time_h"].unique()):
            tr = rna_s[(rna_s["drug"] == drug) & (rna_s["time_h"] == t)]["file_accession"]
            dm = rna_s[(rna_s["drug"] == "DMSO") & (rna_s["time_h"] == t)]["file_accession"]
            controls.append({"drug": drug, "time_h": float(t), "CDKN1A_log2fc": float(rna_k.loc["CDKN1A", tr].mean() - rna_k.loc["CDKN1A", dm].mean())})
    controls = pd.DataFrame(controls)
    report["k562"]["positive_control_cdkn1a"] = {"by_drug_max_log2fc": controls.groupby("drug")["CDKN1A_log2fc"].max().round(3).to_dict(), "passes": bool((controls.groupby("drug")["CDKN1A_log2fc"].max() > 0).all())}
    controls.to_csv(TABLES / "a06_k562_cdkn1a.csv", index=False)

    # ================= T cells =================
    tcell = json.loads((ROOT / "data" / "encode_tcell_manifest.json").read_text())["files"]
    out_t = PROCESSED / "encode_tcell"
    t_rna = [r for r in tcell if r["assay"] == "total RNA-seq"]
    t_dnase = [r for r in tcell if r["assay"] == "DNase-seq"]
    rna_t = rna_table(t_rna, "tcell", symbols)
    rna_t.to_parquet(out_t / "rna_log2fpkm_replicates.parquet")
    grid = pd.read_parquet(PROCESSED / "encode_dex" / "grid.parquet")
    per_bin_t = {r["file_accession"]: grid.assign(value=np.load(out_t / "bins" / f"{r['file_accession']}.npy").astype(float)) for r in t_dnase}
    raw_t = atac.enhancer_openness(training, per_bin_t)
    samples_t = pd.DataFrame([{k: r[k] for k in ("file_accession", "assay", "pair", "donor", "state", "experiment")} | {"rep": rep_number(r)} for r in tcell])
    samples_t.to_csv(out_t / "samples.csv", index=False)
    resting = samples_t[(samples_t["assay"] == "DNase-seq") & (samples_t["state"] == "resting")]["file_accession"]
    perd_values = pd.read_parquet(PERD / "enhancer_openness.parquet").to_numpy().ravel()
    pooled_rest = raw_t[resting].mean(axis=1)
    k = encode_dex.scale_constant(float(np.median(perd_values[perd_values > 0])), float(pooled_rest[pooled_rest > 0].median()))
    openness_t = np.log2(k * raw_t + 1)
    openness_t.to_parquet(out_t / "enhancer_openness_replicates.parquet")
    report["tcell"] = {"rna_replicates": len(t_rna), "dnase_replicates": len(t_dnase), "k": float(k), "usable_gene_coverage": float(np.mean([g in rna_t.index for g in usable]))}

    dn = samples_t[samples_t["assay"] == "DNase-seq"]
    changes, controls_t = {}, {}
    for pair, g in dn.groupby("pair"):
        rest = openness_t[g[g["state"] == "resting"]["file_accession"]].mean(axis=1)
        act = openness_t[g[g["state"] == "activated"]["file_accession"]].mean(axis=1)
        changes[pair] = act - rest
        controls_t[pair] = rest
    noise_t = {}
    for pair, g in dn[dn["state"] == "resting"].groupby("pair"):
        if len(g) >= 2:
            a, b = openness_t[g["file_accession"].iloc[0]], openness_t[g["file_accession"].iloc[1]]
            ok = a.notna() & b.notna() & ((a + b) / 2 >= MIN_MEAN)
            noise_t[pair] = float((a - b)[ok].abs().quantile(0.99))
    report["tcell"]["resting_replicate_noise_p99"] = noise_t
    report["tcell"]["event_threshold"] = float(np.median(list(noise_t.values()))) if noise_t else None
    pairs = sorted(changes)
    for i, j in itertools.permutations(pairs, 2):
        ok = changes[i].notna() & changes[j].notna() & (((controls_t[j] + controls_t[j] + changes[j]) / 2) >= MIN_MEAN)
        tasks.append((("tcell", i, j), changes[i][ok].to_numpy(), changes[j][ok].to_numpy(), deciles(controls_t[j][ok].to_numpy())))

    rs = samples_t[samples_t["assay"] == "total RNA-seq"]
    marker_rows = []
    for pair, g in rs.groupby("pair"):
        act = rna_t[g[g["state"] == "activated"]["file_accession"]].mean(axis=1)
        rest = rna_t[g[g["state"] == "resting"]["file_accession"]].mean(axis=1)
        marker_rows += [{"pair": pair, "gene": m, "log2fc": float(act.get(m, np.nan) - rest.get(m, np.nan))} for m in TCELL_MARKERS]
    markers = pd.DataFrame(marker_rows)
    markers.to_csv(TABLES / "a06_tcell_markers.csv", index=False)
    per_gene = markers.groupby("gene")["log2fc"].apply(lambda v: int((v > 1).sum()))
    report["tcell"]["positive_control"] = {"pairs_with_log2fc_gt_1": per_gene.to_dict(), "passes": bool((per_gene >= 4).all())}

    # ================= gates (parallel) =================
    results = Parallel(n_jobs=n_jobs)(delayed(gate_test)(*task) for task in tasks)
    k562_gate = pd.DataFrame([{"drug": key[1], "time_h": key[2], **r} for key, r in results if key[0] == "k562"])
    k562_gate["group"] = k562_gate["drug"].map(lambda d: "primary" if d in ("Bortezomib", "Chaetocin", "GSK J4", "JQ1") else "sensitivity")
    k562_gate["replicable"] = k562_gate["z"] >= 3
    k562_gate.to_csv(TABLES / "a06_k562_gate.csv", index=False)
    tcell_gate = pd.DataFrame([{"ranking_pair": key[1], "measured_pair": key[2], **r} for key, r in results if key[0] == "tcell"])
    tcell_gate.to_csv(TABLES / "a06_tcell_gate.csv", index=False)
    report["k562"]["gate"] = {"replicable_primary": int(k562_gate.query("group == 'primary'")["replicable"].sum()), "primary_conditions": int((k562_gate["group"] == "primary").sum()), "replicable_sensitivity": int(k562_gate.query("group == 'sensitivity'")["replicable"].sum()), "sensitivity_conditions": int((k562_gate["group"] == "sensitivity").sum())}
    report["tcell"]["gate"] = {"median_cross_pair_z": float(tcell_gate["z"].median()), "passes": bool(tcell_gate["z"].median() >= 3), "n_ordered_pairs": len(tcell_gate)}
    (TABLES / "a06_gates.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(json.dumps(report, indent=2, default=float))
        print(k562_gate.pivot_table(index="drug", columns="time_h", values="z").round(1).to_string())
        print(tcell_gate.pivot_table(index="ranking_pair", columns="measured_pair", values="z").round(1).to_string())
        print(markers.pivot_table(index="pair", columns="gene", values="log2fc").round(2).to_string())


if __name__ == "__main__":
    main()
