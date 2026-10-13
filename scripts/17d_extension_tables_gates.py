"""Amendment 09 B-C: replicate tables, TF activity and PERD-blind gates for the HCT116 degron and SK-N-SH
retinoic-acid series (no PERD output). Mirrors the amendment 06 T-cell processing (13c).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from perd_perturbation import atac, encode_dex, rna, tf_activity

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
MIN_MEAN = 1.0
RA_MARKERS = ["CYP26A1", "RARB", "DHRS3"]
RA_TFS = ["RARA", "RARB", "RARG", "RXRA", "RXRB"]

spec = importlib.util.spec_from_file_location("a06", ROOT / "scripts" / "13c_a06_tables_gates.py")
a06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a06)


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    statuses = {p.stem: json.loads(p.read_text())["status"] for p in (ROOT / "data" / "interim" / "a06_status").glob("*.json")}
    training = a06.training_enhancers()
    symbols = rna.gene_symbol_table(rna.read_gencode_tx2gene(a06.REFERENCE / "gencode.v44.transcripts.fa.gz"))
    usable = sorted({g for genes in pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["usable_genes"] for g in genes})
    grid = pd.read_parquet(PROCESSED / "encode_dex" / "grid.parquet")
    perd_values = pd.read_parquet(PERD / "enhancer_openness.parquet").to_numpy().ravel()
    perd_median = float(np.median(perd_values[perd_values > 0]))
    encode = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    mean, sd = tf_activity.reference_scale(encode)
    raw = pd.read_csv(ROOT / "data" / "raw" / "collectri" / "collectri.tsv", sep="\t")
    weights = tf_activity.weight_matrix(tf_activity.network_from_collectri(raw), mean.index)
    report, tasks, per_unit = {}, [], []
    for dataset in ("hct116", "sknsh"):
        manifest = json.loads((ROOT / "data" / f"encode_{dataset}_manifest.json").read_text())
        records = manifest["files"]
        missing = [r["file_accession"] for r in records if statuses.get(r["file_accession"]) != "done"]
        if missing:
            raise SystemExit(f"{dataset}: download incomplete: {missing[:10]}")
        out = PROCESSED / f"encode_{dataset}"
        rna_records = [r for r in records if r["assay"] != "DNase-seq"]
        dnase_records = [r for r in records if r["assay"] == "DNase-seq"]
        rna_table = a06.rna_table(rna_records, dataset, symbols)
        rna_table.to_parquet(out / "rna_log2fpkm_replicates.parquet")
        activity = tf_activity.activity(rna_table, mean, sd, weights)
        activity.to_parquet(PROCESSED / "tf_activity" / f"activity_{dataset}.parquet")
        per_bin = {r["file_accession"]: grid.assign(value=np.load(out / "bins" / f"{r['file_accession']}.npy").astype(float)) for r in dnase_records}
        raw_open = atac.enhancer_openness(training, per_bin)
        samples = pd.DataFrame([{k: r[k] for k in ("file_accession", "assay", "unit", "state", "experiment")} | {"rep": a06.rep_number(r)} for r in records])
        samples.to_csv(out / "samples.csv", index=False)
        untreated = samples[(samples["assay"] == "DNase-seq") & (samples["state"] == "untreated")]["file_accession"]
        pooled = raw_open[untreated].mean(axis=1)
        k = encode_dex.scale_constant(perd_median, float(pooled[pooled > 0].median()))
        openness = np.log2(k * raw_open + 1)
        openness.to_parquet(out / "enhancer_openness_replicates.parquet")
        rep_info = {"rna_replicates": len(rna_records), "dnase_replicates": len(dnase_records), "k": float(k), "usable_gene_coverage": float(np.mean([g in rna_table.index for g in usable])),
                    "reference_genes_present": int(rna_table.index.isin(mean.index).sum())}

        dn = samples[samples["assay"] == "DNase-seq"]
        noise = {}
        for unit, g in dn[dn["state"] == "untreated"].groupby("unit"):
            g = g.sort_values(["experiment", "rep"])
            if len(g) >= 2:
                a, b = openness[g["file_accession"].iloc[0]], openness[g["file_accession"].iloc[1]]
                ok = a.notna() & b.notna() & ((a + b) / 2 >= MIN_MEAN)
                noise[unit] = float((a - b)[ok].abs().quantile(0.99))
        rep_info["untreated_replicate_noise_p99"] = noise
        rep_info["event_threshold"] = float(np.median(list(noise.values()))) if noise else None
        for unit, g in dn.groupby("unit"):
            t = g[g["state"] == "treated"].sort_values(["experiment", "rep"])["file_accession"].tolist()
            u = g[g["state"] == "untreated"].sort_values(["experiment", "rep"])["file_accession"].tolist()
            row = {"dataset": dataset, "unit": unit, "treated_reps": len(t), "untreated_reps": len(u)}
            if t and u:
                control, treated = openness[u].mean(axis=1), openness[t].mean(axis=1)
                ev = control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)
                change = (treated - control)[ev]
                row.update({"n_evaluable": int(ev.sum()), "change_sd": float(change.std()), "frac_abs_change_gt_threshold": float((change.abs() > (rep_info["event_threshold"] or np.inf)).mean())})
                if len(t) >= 2 and len(u) >= 2:
                    delta_a = (openness[t[0]] - openness[u[0]])[ev].to_numpy()
                    delta_b = (openness[t[1]] - openness[u[1]])[ev].to_numpy()
                    tasks.append(((dataset, unit), delta_a, delta_b, a06.deciles(control[ev].to_numpy())))
            per_unit.append(row)

        rs = samples[samples["assay"] != "DNase-seq"]
        fc_rows = []
        for unit, g in rs.groupby("unit"):
            tr = rna_table[g[g["state"] == "treated"]["file_accession"]].mean(axis=1)
            un = rna_table[g[g["state"] == "untreated"]["file_accession"]].mean(axis=1)
            genes = RA_MARKERS if dataset == "sknsh" else [unit]
            fc_rows += [{"dataset": dataset, "unit": unit, "gene": gene, "log2fc": float(tr.get(gene, np.nan) - un.get(gene, np.nan))} for gene in genes]
            act_t = activity[g[g["state"] == "treated"]["file_accession"]].mean(axis=1)
            act_u = activity[g[g["state"] == "untreated"]["file_accession"]].mean(axis=1)
            diff = (act_t - act_u).sort_values(ascending=False)
            rep_info.setdefault("top_tf_activity_change", {})[unit] = {k_: round(float(v), 3) for k_, v in diff.head(8).items()}
            if dataset == "sknsh":
                rep_info["ra_tf_activity_change"] = {tf: round(float(diff.get(tf, np.nan)), 3) for tf in RA_TFS}
                rep_info["ra_tf_activity_rank"] = {tf: int(list(diff.index).index(tf) + 1) for tf in RA_TFS if tf in diff.index}
        fc = pd.DataFrame(fc_rows)
        rep_info["rna_markers"] = fc.to_dict("records")
        if dataset == "sknsh":
            rep_info["positive_control_passes"] = bool((fc["log2fc"] > 1).sum() >= 2)
        report[dataset] = rep_info

    results = Parallel(n_jobs=n_jobs)(delayed(a06.gate_test)(*task) for task in tasks)
    gate = {f"{k[0]}|{k[1]}": r for k, r in results}
    units = pd.DataFrame(per_unit)
    units["gate_z"] = [gate.get(f"{d}|{u}", {}).get("z", np.nan) for d, u in zip(units["dataset"], units["unit"])]
    units["replicable"] = units["gate_z"] > 3
    units.to_csv(TABLES / "a09_gate_per_unit.csv", index=False)
    report["gate"] = units.to_dict("records")
    (TABLES / "a09_gates.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    with pd.option_context("display.width", 220):
        print(units.round(3).to_string(index=False))
        print(json.dumps({d: {k: v for k, v in r.items() if k not in ("rna_markers",)} for d, r in report.items() if d != "gate"}, indent=1, default=float))
        print(pd.DataFrame(report["sknsh"]["rna_markers"] + report["hct116"]["rna_markers"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
