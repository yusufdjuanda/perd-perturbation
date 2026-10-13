"""Amendment 05 A (primary): PERD's own differential-enhancer pipeline on ENCODE dexamethasone, validated
against measured DNase change. Candidate filters F1-F3, limma calls (paper thresholds and sensitivities),
stratified enrichment of measured opening among PERD up-calls. Definitions fixed in execution.md first.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from perd_perturbation import evaluation
from perd_perturbation import variant
from perd_perturbation.vendor.perd_repro.cell_types import extract_cell_type


ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
OUT = ROOT / "data" / "processed" / "encode_dex"
LIMMA_DIR = variant.tagged(ROOT / "data" / "interim" / "encode_dex_limma")
TABLES = ROOT / "results" / "tables"
MANIFEST = ROOT / "data" / "encode_dex_manifest.json"
EVENT_THRESHOLD = 1.1292896530999152
MIN_MEAN = 1.0
F3_COUNT = 9_340
MIN_CALLS = 20


def candidate_sets(fitted: pd.Index, rna_symbols: set[str]) -> dict[str, pd.Index]:
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"]
    openness = openness.loc[training]
    spread = (openness > 0).sum(axis=1)
    sd = openness.std(axis=1)
    cell_type = pd.Series({c: extract_cell_type(c) for c in openness.columns})
    by_type = (openness > 2).T.groupby(cell_type).any().T
    specificity = by_type.sum(axis=1)
    sd_cut = sd.quantile(0.20)
    f1 = openness.index[(spread > 100) & (sd >= sd_cut) & (specificity > 10)]
    genehancer = pd.read_parquet(PERD / "genehancer_enhancers.parquet").set_index("enhancer_id")
    tf = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"]
    n_tg = genehancer.loc[training, "target_genes"].map(lambda g: sum(x in rna_symbols for x in g))
    n_tf = tf.reindex(training).map(lambda g: sum(x in rna_symbols for x in g) if isinstance(g, (list, np.ndarray)) else 0)
    base = training[(n_tg.to_numpy() >= 2) & (n_tf.to_numpy() >= 2)]
    f2 = pd.Index(sorted(set(base) & set(f1)))
    pre_f3 = pd.Index(sorted(set(base) & set(openness.index[(spread > 100) & (specificity > 10)])))
    f3 = sd.loc[pre_f3].sort_values(ascending=False).index[:F3_COUNT]
    counts = {"cell_types": int(cell_type.nunique()), "sd_20th_percentile": float(sd_cut), "F1_before_fit": len(f1), "paper_base_before_fit": len(base), "F2_before_fit": len(f2), "F3_before_fit": len(f3)}
    sets = {name: pd.Index(sorted(set(s) & set(fitted))) for name, s in (("F1", f1), ("F2", f2), ("F3", f3))}
    counts.update({f"{k}_fitted": len(v) for k, v in sets.items()})
    return sets, counts


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    files = pd.DataFrame(json.loads(MANIFEST.read_text())["files"]).sort_values(["time_h", "biological_replicate"])
    dnase = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "DNase"].groupby("time_h")}
    rna_files = files[files["assay"] == "RNA"]
    samples = rna_files[["file_accession", "time_h"]].rename(columns={"file_accession": "accession"})
    times = sorted(t for t in dnase if t != 0)
    predictions = {m: variant.read_predictions(OUT, m) for m in ("rf", "ridge")}
    fitted = predictions["rf"].index[predictions["rf"]["skip_reason"] == ""]
    rna_symbols = set(pd.read_parquet(OUT / "rna_log2fpkm_replicates.parquet").index)
    sets, counts = candidate_sets(fitted, rna_symbols)
    print(json.dumps(counts, indent=2), flush=True)

    measured = pd.read_parquet(OUT / "enhancer_openness_replicates.parquet")
    control_mean = measured[dnase[0]].mean(axis=1)
    LIMMA_DIR.mkdir(parents=True, exist_ok=True)
    samples.to_csv(LIMMA_DIR / "samples.tsv", sep="\t", index=False)

    rows = []
    for model in ("rf", "ridge"):
        for set_name, enhancers in sets.items():
            out_dir = LIMMA_DIR / f"{model}_{set_name}"
            matrix_path = LIMMA_DIR / f"{model}_{set_name}_matrix.tsv"
            predictions[model].loc[enhancers, samples["accession"]].to_csv(matrix_path, sep="\t", index_label="enhancer_id")
            result = subprocess.run(["pixi", "run", "-e", "r", "Rscript", str(ROOT / "scripts" / "R" / "limma_diff.R"), str(matrix_path), str(LIMMA_DIR / "samples.tsv"), str(out_dir)], check=True, capture_output=True, text=True)
            print(model, set_name, result.stdout.strip(), flush=True)
            matrix_path.unlink()
            for t in times:
                table = pd.read_csv(out_dir / f"limma_t{t:g}.tsv", sep="\t").set_index("enhancer_id")
                treated_mean = measured.loc[table.index, dnase[t]].mean(axis=1)
                ctrl = control_mean.loc[table.index]
                change = treated_mean - ctrl
                evaluable = ((treated_mean + ctrl) / 2 >= MIN_MEAN) & change.notna()
                tab = table.loc[evaluable]
                ch = change.loc[evaluable]
                stratum = pd.qcut(ctrl.loc[evaluable].rank(method="first"), 10, labels=False).to_numpy()
                opening, closing = (ch > EVENT_THRESHOLD).to_numpy(), (ch < -EVENT_THRESHOLD).to_numpy()
                t_cut_up = tab["t"].quantile(0.99)
                t_cut_down = tab["t"].quantile(0.01)
                rules = {
                    "paper": ((tab["logFC"] > 0.5) & (tab["P.Value"] < 0.001), (tab["logFC"] < -0.5) & (tab["P.Value"] < 0.001)),
                    "bh_fdr_0.05": ((tab["logFC"] > 0) & (tab["adj.P.Val"] < 0.05), (tab["logFC"] < 0) & (tab["adj.P.Val"] < 0.05)),
                    "top1pct_t": (tab["t"] >= t_cut_up, tab["t"] <= t_cut_down),
                }
                for rule, (up, down) in rules.items():
                    for direction, calls, outcome in (("opening", up.to_numpy(), opening), ("closing", down.to_numpy(), closing)):
                        cmh = evaluation.cmh_odds_ratio(calls, outcome, stratum)
                        rows.append({
                            "model": model, "candidate_set": set_name, "rule": rule, "direction": direction, "time_h": t,
                            "n_candidates_evaluable": int(evaluable.sum()), "n_calls": int(calls.sum()),
                            "n_events": int(outcome.sum()), "precision": float(outcome[calls].mean()) if calls.any() else np.nan,
                            "event_rate_all": float(outcome.mean()), "recall": float(calls[outcome].mean()) if outcome.any() else np.nan,
                            **{k: v for k, v in cmh.items() if k not in ("exposed", "exposed_with_outcome")},
                        })
    table = pd.DataFrame(rows)
    table["p_holm"] = np.nan
    decision = []
    for keys, group in table.groupby(["model", "candidate_set", "rule", "direction"]):
        adjusted = evaluation.holm(group["p_one_sided"].to_numpy())
        table.loc[group.index, "p_holm"] = adjusted
        enough = group["n_calls"].to_numpy() >= MIN_CALLS
        validated = enough & (group["odds_ratio"].to_numpy() > 1) & (adjusted < 0.05)
        decision.append({
            "model": keys[0], "candidate_set": keys[1], "rule": keys[2], "direction": keys[3],
            "time_points_validated": int(validated.sum()), "time_points_too_few_calls": int((~enough).sum()),
            "overall_validated": bool(validated.sum() >= 6),
            "median_calls": float(group["n_calls"].median()), "median_odds_ratio": float(group["odds_ratio"].median()),
            "median_precision": float(group["precision"].median()), "median_event_rate": float(group["event_rate_all"].median()),
        })
    table.to_csv(variant.tagged(TABLES / "a05_perd_pipeline_per_time.csv"), index=False)
    decision = pd.DataFrame(decision)
    decision.to_csv(variant.tagged(TABLES / "a05_perd_pipeline_decision.csv"), index=False)
    variant.tagged(TABLES / "a05_candidate_sets.json").write_text(json.dumps(counts, indent=2) + "\n")
    with pd.option_context("display.width", 220, "display.max_rows", 100):
        print(decision.round(3).to_string(index=False))
        primary = table[(table["candidate_set"] == "F1") & (table["rule"] == "paper") & (table["direction"] == "opening")]
        print(primary[["model", "time_h", "n_calls", "n_events", "precision", "event_rate_all", "odds_ratio", "ci_low", "ci_high", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
