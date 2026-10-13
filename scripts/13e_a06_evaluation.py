"""Amendment 06 D: PERD evaluation on the K562 drug panel and T-cell activation, plus the three-dataset comparison.

Definitions fixed in runbook_amendment_06.md / execution.md before any data was downloaded.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from perd_perturbation import evaluation, plate_eval
from perd_perturbation import variant

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
LIMMA = variant.tagged(ROOT / "data" / "interim" / "a06_limma")
MIN_MEAN = 1.0
MIN_CALLS = 20
MODELS = ("rf", "ridge")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_module("pipeline", ROOT / "scripts" / "11b_perd_pipeline_dex.py")
nspe = load_module("nspe", ROOT / "scripts" / "09h_nspe.py")


def extreme(key, predicted, measured, stratum):
    k = plate_eval.k_values(len(predicted))[0.01]
    return key, plate_eval.extreme_test_components(predicted, measured, stratum, {0.01: k}, n_perm=2000, seed=42)[0.01]


def side_rows(label: dict, measured: dict[str, pd.Series], predicted: dict[str, dict[str, pd.Series]], members: dict[str, pd.Index]) -> list[dict]:
    rows = []
    for set_name, idx in members.items():
        for side in ("control", "treated", "change"):
            m = measured[side].reindex(idx)
            for model in MODELS:
                p = predicted[model][side].reindex(idx)
                ok = m.notna() & p.notna()
                if side == "change":
                    ok &= measured["evaluable"].reindex(idx).fillna(False)
                result = nspe.metrics(m[ok].to_numpy(), p[ok].to_numpy())
                rows.append({**label, "set": set_name, "side": side, "model": model, "spearman": evaluation.correlation(m[ok], p[ok]), **result})
    return rows


def bootstrap_units(values: pd.DataFrame, unit: str, level: float = 0.975, n_boot: int = 10_000) -> dict:
    """Mean z over conditions with a bootstrap over units (drugs or pairs)."""
    units = values[unit].unique()
    rng = np.random.default_rng(42)
    per_unit = {u: values.loc[values[unit] == u, "z"].to_numpy() for u in units}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(units, len(units), replace=True)
        draws.append(np.concatenate([per_unit[u] for u in pick]).mean())
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(draws, [tail, 100 - tail])
    return {"mean_z": float(values["z"].mean()), "ci_low": float(low), "ci_high": float(high), "level": level, "n_units": len(units), "n_conditions": len(values)}


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    gates = json.loads((TABLES / "a06_gates.json").read_text())
    report: dict = {"gates": gates}
    sides, tasks, meta = [], [], {}

    # ================= K562 =================
    out = PROCESSED / "encode_k562"
    samples = pd.read_csv(out / "samples.csv")
    openness = pd.read_parquet(out / "enhancer_openness_replicates.parquet")
    preds = {m: variant.read_predictions(out, m) for m in MODELS}
    fitted = preds["rf"].index[preds["rf"]["skip_reason"] == ""]
    rna_symbols = set(pd.read_parquet(out / "rna_log2fpkm_replicates.parquet").index)
    candidates, counts = pipeline.candidate_sets(fitted, rna_symbols)
    report["k562_candidate_counts"] = counts
    gate = pd.read_csv(TABLES / "a06_k562_gate.csv")
    replicable = gate[gate["replicable"]]
    atac_s, rna_s = samples[samples["assay"] == "ATAC-seq"], samples[samples["assay"] == "total RNA-seq"]
    n_genes = preds["rf"]["n_genes_used"]
    noise = {float(k): v["event_threshold_p99"] for k, v in gates["k562"]["dmso_noise_by_time"].items()}

    # limma per time point and model (controls = DMSO RNA replicates of that time; codes 1..n = drugs).
    LIMMA.mkdir(parents=True, exist_ok=True)
    limma_tables = {}
    for t in sorted(replicable["time_h"].unique()):
        dmso = rna_s[(rna_s["drug"] == "DMSO") & (rna_s["time_h"] == t)]["file_accession"].tolist()
        drugs = sorted(replicable[replicable["time_h"] == t]["drug"])
        sheet = [{"accession": a, "time_h": 0} for a in dmso]
        for code, drug in enumerate(drugs, start=1):
            sheet += [{"accession": a, "time_h": code} for a in rna_s[(rna_s["drug"] == drug) & (rna_s["time_h"] == t)]["file_accession"]]
        sheet = pd.DataFrame(sheet)
        sheet_path = LIMMA / f"samples_t{t:g}.tsv"
        sheet.to_csv(sheet_path, sep="\t", index=False)
        for model in MODELS:
            matrix = LIMMA / f"{model}_t{t:g}_matrix.tsv"
            preds[model].loc[candidates["F1"], sheet["accession"]].to_csv(matrix, sep="\t", index_label="enhancer_id")
            out_dir = LIMMA / f"{model}_t{t:g}"
            subprocess.run(["pixi", "run", "-e", "r", "Rscript", str(ROOT / "scripts" / "R" / "limma_diff.R"), str(matrix), str(sheet_path), str(out_dir)], check=True, capture_output=True, text=True)
            matrix.unlink()
            for code, drug in enumerate(drugs, start=1):
                limma_tables[(model, drug, float(t))] = pd.read_csv(out_dir / f"limma_t{code}.tsv", sep="\t").set_index("enhancer_id")

    pipeline_rows = []
    for row in replicable.itertuples():
        drug, t, group = row.drug, float(row.time_h), row.group
        dmso_atac = atac_s[(atac_s["drug"] == "DMSO") & (atac_s["time_h"] == t)]["file_accession"]
        trt_atac = atac_s[(atac_s["drug"] == drug) & (atac_s["time_h"] == t)]["file_accession"]
        dmso_rna = rna_s[(rna_s["drug"] == "DMSO") & (rna_s["time_h"] == t)]["file_accession"]
        trt_rna = rna_s[(rna_s["drug"] == drug) & (rna_s["time_h"] == t)]["file_accession"]
        control, treated = openness[dmso_atac].mean(axis=1), openness[trt_atac].mean(axis=1)
        change = treated - control
        evaluable = control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)
        idx = evaluable[evaluable].index.intersection(fitted)
        p = {m: {"control": preds[m].loc[fitted, dmso_rna].mean(axis=1), "treated": preds[m].loc[fitted, trt_rna].mean(axis=1)} for m in MODELS}
        for m in MODELS:
            p[m]["change"] = p[m]["treated"] - p[m]["control"]
            stratum = plate_eval.strata(control.loc[idx].to_numpy(), n_genes.loc[idx].to_numpy())
            tasks.append((("k562", drug, t, m), p[m]["change"].loc[idx].to_numpy(), change.loc[idx].to_numpy(), stratum))
            meta[("k562", drug, t, m)] = {"group": group}
            # PERD's own pipeline (F1, paper thresholds) vs measured opening above the time's DMSO noise.
            tab = limma_tables[(m, drug, t)]
            cand = tab.index.intersection(idx)
            ch = change.loc[cand]
            calls_up = ((tab.loc[cand, "logFC"] > 0.5) & (tab.loc[cand, "P.Value"] < 0.001)).to_numpy()
            outcome = (ch > noise[t]).to_numpy()
            strat = pd.qcut(control.loc[cand].rank(method="first"), 10, labels=False).to_numpy()
            cmh = evaluation.cmh_odds_ratio(calls_up, outcome, strat)
            pipeline_rows.append({"drug": drug, "time_h": t, "group": group, "model": m, "n_calls": int(calls_up.sum()), "n_events": int(outcome.sum()), "precision": float(outcome[calls_up].mean()) if calls_up.any() else np.nan, "event_rate": float(outcome.mean()), **{k: v for k, v in cmh.items() if k not in ("exposed", "exposed_with_outcome")}})
        sides += side_rows({"dataset": "K562", "condition": f"{drug} {t:g}h", "group": group}, {"control": control, "treated": treated, "change": change, "evaluable": evaluable}, p, {"all": fitted, "F1": candidates["F1"]})
    pipe = pd.DataFrame(pipeline_rows)
    pipe["p_holm"] = np.nan
    for (group, model), g in pipe.groupby(["group", "model"]):
        pipe.loc[g.index, "p_holm"] = evaluation.holm(g["p_one_sided"].to_numpy())
    pipe["validated"] = (pipe["odds_ratio"] > 1) & (pipe["p_holm"] < 0.05) & (pipe["n_calls"] >= MIN_CALLS)
    pipe.to_csv(variant.tagged(TABLES / "a06_k562_pipeline.csv"), index=False)

    # ================= T cells =================
    out_t = PROCESSED / "encode_tcell"
    samples_t = pd.read_csv(out_t / "samples.csv")
    openness_t = pd.read_parquet(out_t / "enhancer_openness_replicates.parquet")
    preds_t = {m: variant.read_predictions(out_t, m) for m in MODELS}
    fitted_t = preds_t["rf"].index[preds_t["rf"]["skip_reason"] == ""]
    candidates_t, _ = pipeline.candidate_sets(fitted_t, set(pd.read_parquet(out_t / "rna_log2fpkm_replicates.parquet").index))
    tcell_ok = gates["tcell"]["gate"]["passes"]
    if tcell_ok:
        for pair, g in samples_t.groupby("pair"):
            dn, rs = g[g["assay"] == "DNase-seq"], g[g["assay"] == "total RNA-seq"]
            control = openness_t[dn[dn["state"] == "resting"]["file_accession"]].mean(axis=1)
            treated = openness_t[dn[dn["state"] == "activated"]["file_accession"]].mean(axis=1)
            change = treated - control
            evaluable = control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)
            idx = evaluable[evaluable].index.intersection(fitted_t)
            p = {m: {"control": preds_t[m].loc[fitted_t, rs[rs["state"] == "resting"]["file_accession"]].mean(axis=1), "treated": preds_t[m].loc[fitted_t, rs[rs["state"] == "activated"]["file_accession"]].mean(axis=1)} for m in MODELS}
            for m in MODELS:
                p[m]["change"] = p[m]["treated"] - p[m]["control"]
                stratum = plate_eval.strata(control.loc[idx].to_numpy(), preds_t["rf"]["n_genes_used"].loc[idx].to_numpy())
                tasks.append((("tcell", pair, 36.0, m), p[m]["change"].loc[idx].to_numpy(), change.loc[idx].to_numpy(), stratum))
                meta[("tcell", pair, 36.0, m)] = {"group": "tcell"}
            sides += side_rows({"dataset": "T cell", "condition": pair, "group": "tcell"}, {"control": control, "treated": treated, "change": change, "evaluable": evaluable}, p, {"all": fitted_t, "F1": candidates_t["F1"]})

    results = Parallel(n_jobs=n_jobs)(delayed(extreme)(*task) for task in tasks)
    ext = pd.DataFrame([{"dataset": k[0], "unit": k[1], "time_h": k[2], "model": k[3], **meta[k], **r} for k, r in results])
    ext.to_csv(variant.tagged(TABLES / "a06_extreme_per_condition.csv"), index=False)
    side_table = pd.DataFrame(sides)
    side_table.to_csv(variant.tagged(TABLES / "a06_nspe_sides.csv"), index=False)

    summary = {}
    for (dataset, group, model), g in ext.groupby(["dataset", "group", "model"]):
        summary[f"{dataset}|{group}|{model}"] = bootstrap_units(g, "unit")
    report["extreme_summary"] = summary
    report["k562_pipeline_summary"] = pipe.groupby(["group", "model"]).agg(validated=("validated", "sum"), conditions=("validated", "size"), median_or=("odds_ratio", "median"), median_precision=("precision", "median"), median_calls=("n_calls", "median")).reset_index().to_dict("records")

    # ================= three-dataset comparison =================
    comparison = []
    dex = pd.read_csv(variant.tagged(TABLES / "encode_dex_eval_decision.csv")).query("subset == 'all' and metric == 'mean_z_over_11_times'").set_index("predictor")
    dex_pipe = pd.read_csv(variant.tagged(TABLES / "a05_perd_pipeline_decision.csv")).query("candidate_set == 'F1' and rule == 'paper' and direction == 'opening'").set_index("model")
    dex_change = pd.read_csv(variant.tagged(TABLES / "subset_change_metrics_per_time.csv")).query("set == 'F1'").groupby("model")["nspe"].median()
    dex_sides = pd.read_csv(variant.tagged(TABLES / "before_after_metrics.csv")).query("set == 'F1' and filter == 'defined'").groupby("model")["nspe"].median()
    for model in MODELS:
        name = f"perd_{model}"
        comparison.append({"dataset": "A549 dexamethasone", "model": model, "extreme_mean_z": dex.loc[name, "observed"], "extreme_ci": f"[{dex.loc[name, 'ci_low']:.2f}, {dex.loc[name, 'ci_high']:.2f}]", "pipeline_validated": f"{int(dex_pipe.loc[model, 'time_points_validated'])}/11", "pipeline_median_or": dex_pipe.loc[model, "median_odds_ratio"], "change_nspe_median_F1": dex_change.loc[model], "side_nspe_median_F1": dex_sides.loc[model]})
        for group, label in (("primary", "K562 primary (4 drugs)"), ("sensitivity", "K562 sensitivity (7 drugs)")):
            e = summary.get(f"K562|{group}|{model}") or summary.get(f"k562|{group}|{model}")
            pg = pipe[(pipe["group"] == group) & (pipe["model"] == model)]
            sg = side_table[(side_table["dataset"] == "K562") & (side_table["group"] == group) & (side_table["model"] == model) & (side_table["set"] == "F1")]
            comparison.append({"dataset": label, "model": model, "extreme_mean_z": e["mean_z"] if e else np.nan, "extreme_ci": f"[{e['ci_low']:.2f}, {e['ci_high']:.2f}]" if e else "", "pipeline_validated": f"{int(pg['validated'].sum())}/{len(pg)}", "pipeline_median_or": pg["odds_ratio"].median(), "change_nspe_median_F1": sg[sg["side"] == "change"]["nspe"].median(), "side_nspe_median_F1": sg[sg["side"] != "change"]["nspe"].median()})
        e = summary.get(f"tcell|tcell|{model}")
        sg = side_table[(side_table["dataset"] == "T cell") & (side_table["model"] == model) & (side_table["set"] == "F1")]
        comparison.append({"dataset": "T-cell activation (5 donor pairs)", "model": model, "extreme_mean_z": e["mean_z"] if e else np.nan, "extreme_ci": f"[{e['ci_low']:.2f}, {e['ci_high']:.2f}]" if e else "", "pipeline_validated": "not possible (1 replicate)", "pipeline_median_or": np.nan, "change_nspe_median_F1": sg[sg["side"] == "change"]["nspe"].median() if len(sg) else np.nan, "side_nspe_median_F1": sg[sg["side"] != "change"]["nspe"].median() if len(sg) else np.nan})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(variant.tagged(TABLES / "a06_three_dataset_comparison.csv"), index=False)
    variant.tagged(TABLES / "a06_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(comparison.round(3).to_string(index=False))
        print(pipe[["drug", "time_h", "group", "model", "n_calls", "precision", "event_rate", "odds_ratio", "p_holm", "validated"]].round(4).to_string(index=False))
        print(ext.pivot_table(index=["dataset", "unit", "time_h"], columns="model", values="z").round(2).to_string())


if __name__ == "__main__":
    main()
