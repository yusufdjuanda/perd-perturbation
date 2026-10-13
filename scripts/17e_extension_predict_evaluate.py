"""Amendment 09 D: frozen PERD-RF / PERD-ridge predictions for the HCT116 and SK-N-SH RNA replicates (one pass, as 13d),
and the replication evaluation for HCT116 degrons only (SK-N-SH treated is in PERD's training cells: not a test).
Extreme test per replicable unit (K = 1%), mean z with a bootstrap over units, NSPE before / after / change (F1).
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from perd_perturbation import plate_eval
from perd_perturbation.perd_rf import run_perd_model

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
MODELS = ("rf", "ridge")
MIN_MEAN = 1.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a06 = load_module("a06eval", ROOT / "scripts" / "13e_a06_evaluation.py")


def predict(n_jobs: int) -> dict:
    tables = {d: pd.read_parquet(PROCESSED / f"encode_{d}" / "rna_log2fpkm_replicates.parquet") for d in ("hct116", "sknsh")}
    genes = tables["hct116"].index.union(tables["sknsh"].index)
    profiles = pd.concat([t.reindex(genes).fillna(0.0) for t in tables.values()], axis=1)
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    encode_rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    summary = {"profiles": int(profiles.shape[1])}
    for model in MODELS:
        start = time.time()
        table = run_perd_model(training, encode_rna, openness, profiles, model=model, n_jobs=n_jobs)
        fitted = table.index[table["skip_reason"] == ""]
        chosen = np.random.default_rng(0).choice(fitted.to_numpy(), size=200, replace=False)
        rerun = run_perd_model(training[training["enhancer_id"].isin(chosen)], encode_rna, openness, profiles, model=model, n_jobs=n_jobs)
        identical = bool(np.array_equal(rerun[list(profiles.columns)].to_numpy(), table.loc[rerun.index, list(profiles.columns)].to_numpy()))
        meta = [c for c in ("n_genes_train", "n_genes_used", "skip_reason", "alpha") if c in table.columns]
        for dataset, t in tables.items():
            table[list(t.columns) + meta].to_parquet(PROCESSED / f"encode_{dataset}" / f"perd_predictions_{model}.parquet")
        summary[model] = {"fitted": len(fitted), "minutes": (time.time() - start) / 60, "determinism_identical": identical}
        print(model, summary[model], flush=True)
        if not identical:
            raise SystemExit(f"{model}: determinism check FAILED")
    return summary


def evaluate(n_jobs: int) -> dict:
    out = PROCESSED / "encode_hct116"
    samples = pd.read_csv(out / "samples.csv")
    openness = pd.read_parquet(out / "enhancer_openness_replicates.parquet")
    preds = {m: pd.read_parquet(out / f"perd_predictions_{m}.parquet") for m in MODELS}
    fitted = preds["rf"].index[preds["rf"]["skip_reason"] == ""]
    candidates, _ = a06.pipeline.candidate_sets(fitted, set(pd.read_parquet(out / "rna_log2fpkm_replicates.parquet").index))
    gate = pd.read_csv(TABLES / "a09_gate_per_unit.csv")
    replicable = gate[(gate["dataset"] == "hct116") & gate["replicable"]]["unit"].tolist()
    tasks, sides = [], []
    for unit in replicable:
        g = samples[samples["unit"] == unit]
        dn, rs = g[g["assay"] == "DNase-seq"], g[g["assay"] != "DNase-seq"]
        control = openness[dn[dn["state"] == "untreated"]["file_accession"]].mean(axis=1)
        treated = openness[dn[dn["state"] == "treated"]["file_accession"]].mean(axis=1)
        change = treated - control
        evaluable = control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)
        idx = evaluable[evaluable].index.intersection(fitted)
        p = {m: {"control": preds[m].loc[fitted, rs[rs["state"] == "untreated"]["file_accession"]].mean(axis=1),
                 "treated": preds[m].loc[fitted, rs[rs["state"] == "treated"]["file_accession"]].mean(axis=1)} for m in MODELS}
        for m in MODELS:
            p[m]["change"] = p[m]["treated"] - p[m]["control"]
            stratum = plate_eval.strata(control.loc[idx].to_numpy(), preds["rf"]["n_genes_used"].loc[idx].to_numpy())
            tasks.append((("hct116", unit, 6.0, m), p[m]["change"].loc[idx].to_numpy(), change.loc[idx].to_numpy(), stratum))
        sides += a06.side_rows({"dataset": "HCT116", "condition": unit, "group": "degron"}, {"control": control, "treated": treated, "change": change, "evaluable": evaluable}, p, {"all": fitted, "F1": candidates["F1"]})
    results = Parallel(n_jobs=n_jobs)(delayed(a06.extreme)(*task) for task in tasks)
    ext = pd.DataFrame([{"dataset": k[0], "unit": k[1], "time_h": k[2], "model": k[3], "group": "degron", **r} for k, r in results])
    ext.to_csv(TABLES / "a09_hct116_extreme_per_unit.csv", index=False)
    side_table = pd.DataFrame(sides)
    side_table.to_csv(TABLES / "a09_hct116_nspe_sides.csv", index=False)
    summary = {"replicable_units": replicable}
    for m in MODELS:
        e = ext[ext["model"] == m]
        summary[f"extreme_{m}"] = a06.bootstrap_units(e, "unit") if len(e) else None
        s = side_table[(side_table["model"] == m) & (side_table["set"] == "F1")]
        summary[f"nspe_F1_{m}"] = {side: float(s[s["side"] == side]["nspe"].median()) for side in ("control", "treated", "change")} if len(s) else None
    with pd.option_context("display.width", 200):
        print(ext.pivot_table(index="unit", columns="model", values="z").round(2).to_string() if len(ext) else "no replicable units")
        print(json.dumps(summary, indent=1, default=float))
    return summary


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    report = {"predictions": predict(n_jobs)}
    report["hct116_evaluation"] = evaluate(n_jobs)
    (TABLES / "a09_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
