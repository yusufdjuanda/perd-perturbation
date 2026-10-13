"""Amendment 03 section E with amendment 04: frozen PERD vs measured DNase change, ENCODE A549 dexamethasone.

All definitions were fixed in execution.md (2026-10-04, "section E ... definitions fixed") before this ran.
Amendment 07 D: with PERD_VARIANT=<tag>, perd_rf/perd_ridge are the variant's predictions, frozen PERD is scored
alongside as frozen_rf/frozen_ridge on the same bootstrap draws, paired differences are added, and outputs get _<tag>.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import sparse  # noqa: E402

from perd_perturbation import evaluation, plate_eval, variant  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
OUT = ROOT / "data" / "processed" / "encode_dex"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MANIFEST = ROOT / "data" / "encode_dex_manifest.json"
MIN_MEAN = 1.0
EVENT_THRESHOLD = 1.1292896530999152  # amendment 03 C3, 99th percentile of control-half noise
PRIMARY_K = 0.01
N_BOOT = 2000
MODELS = ("perd_rf", "perd_ridge")
FROZEN = ("frozen_rf", "frozen_ridge") if variant.tag() else ()
RNA_PREDICTORS = ("perd_rf", "perd_ridge", "gene_mean_fc", *FROZEN)
FIXED_PREDICTORS = ("encode_mean_openness_prior", "random_ranking", "control_openness_diagnostic")


def load():
    files = pd.DataFrame(json.loads(MANIFEST.read_text())["files"]).sort_values(["time_h", "biological_replicate"])
    dnase = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "DNase"].groupby("time_h")}
    rna = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "RNA"].groupby("time_h")}
    measured = pd.read_parquet(OUT / "enhancer_openness_replicates.parquet")
    predictions = {m: variant.read_predictions(OUT, m.split("_")[1]) for m in MODELS}
    frozen = {f"frozen_{m}": variant.read_predictions(OUT, m, variant="") for m in ("rf", "ridge")} if FROZEN else {}
    fitted = predictions["perd_rf"].index[predictions["perd_rf"]["skip_reason"] == ""]
    assert fitted.equals(predictions["perd_ridge"].index[predictions["perd_ridge"]["skip_reason"] == ""])
    profiles = pd.read_parquet(OUT / "rna_log2fpkm_replicates.parquet")
    usable = pd.read_parquet(PERD / "netopen_training_enhancers.parquet").set_index("enhancer_id")["usable_genes"]
    gene_index = {g: i for i, g in enumerate(profiles.index)}
    rows, cols = [], []
    for i, e in enumerate(fitted):
        used = [gene_index[g] for g in usable[e] if g in gene_index]
        rows += [i] * len(used)
        cols += used
    incidence = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(fitted), len(profiles)))
    incidence = sparse.diags(1 / np.asarray(incidence.sum(axis=1)).ravel()) @ incidence
    gene_mean = pd.DataFrame(incidence @ profiles.to_numpy(), index=fitted, columns=profiles.columns)
    per_replicate = {"perd_rf": predictions["perd_rf"].loc[fitted, profiles.columns], "perd_ridge": predictions["perd_ridge"].loc[fitted, profiles.columns], "gene_mean_fc": gene_mean}
    per_replicate.update({name: table.loc[fitted, profiles.columns] for name, table in frozen.items()})
    perd_openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    encode_prior = perd_openness.mean(axis=1).reindex(fitted)
    sd = perd_openness.std(axis=1)
    training_ids = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")["enhancer_id"]
    variable = (sd.reindex(training_ids) >= sd.reindex(training_ids).median()).reindex(fitted).fillna(False)
    return dnase, rna, measured.loc[fitted], per_replicate, predictions["perd_rf"].loc[fitted, "n_genes_used"], encode_prior, variable


def observed_test(key: tuple, vector: np.ndarray, measured: np.ndarray, stratum: np.ndarray, ks: dict) -> list[dict]:
    t, subset_name, name = key
    result = plate_eval.extreme_test_components(vector, measured, stratum, ks, n_perm=2000, seed=42)
    spearman = evaluation.correlation(vector, measured)
    pearson = evaluation.correlation(vector, measured, "pearson")
    return [{"time_h": t, "subset": subset_name, "predictor": name, "k_fraction": fraction, **r, "spearman": spearman, "pearson": pearson, "n_evaluable": len(vector)} for fraction, r in result.items()]


def bootstrap_chunk(seed_sequences, data: dict) -> list[dict]:
    """Replicate bootstrap draws: resample DNase and RNA replicates within each condition (control once per draw)."""
    m_col = {a: i for i, a in enumerate(data["measured_columns"])}
    r_col = {a: i for i, a in enumerate(data["replicate_columns"])}
    out = []
    for seed_sequence in seed_sequences:
        rng = np.random.default_rng(seed_sequence)

        def pick(accessions):
            return [accessions[i] for i in rng.integers(0, len(accessions), len(accessions))]

        def m_mean(accessions):
            return data["measured"][:, [m_col[a] for a in accessions]].mean(axis=1)

        def r_mean(name, accessions):
            return data["replicates"][name][:, [r_col[a] for a in accessions]].mean(axis=1)

        m_ctrl = m_mean(pick(data["dnase"][0]))
        rna_ctrl = pick(data["rna"][0])
        p_ctrl = {name: r_mean(name, rna_ctrl) for name in data["rna_predictors"]}
        z = {}
        for index, t in enumerate(data["times"]):
            m = m_mean(pick(data["dnase"][t])) - m_ctrl
            rna_t = pick(data["rna"][t])
            vectors = {name: r_mean(name, rna_t) - p_ctrl[name] for name in data["rna_predictors"]}
            vectors.update(data["fixed"])
            vectors["random_ranking"] = np.random.default_rng(42 + index).standard_normal(len(m))
            for subset_name in ("all", "variable"):
                mask, k = data["setup"][(t, subset_name)]["mask"], data["setup"][(t, subset_name)]["k"]
                for name, v in vectors.items():
                    stat, _, _ = plate_eval.extreme_statistics(v[mask], m[mask], k)
                    mean_null, sd_null = data["null_moments"][(t, subset_name, name)]
                    z[(t, subset_name, name, "z")] = (stat - mean_null) / sd_null
                    if subset_name == "all" or name in data["models"]:
                        z[(t, subset_name, name, "spearman")] = evaluation.correlation(v[mask], m[mask])
        out.append(z)
    return out


def changes(table: pd.DataFrame, treated: list[str], control: list[str]) -> np.ndarray:
    return table[treated].to_numpy().mean(axis=1) - table[control].to_numpy().mean(axis=1)


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    dnase, rna, measured, per_replicate, n_genes, encode_prior, variable = load()
    times = sorted(t for t in dnase if t != 0)
    control_mean = measured[dnase[0]].mean(axis=1).to_numpy()
    enhancers = measured.index
    reliability = pd.read_csv(TABLES / "encode_dex_reliability.csv").query("scale == 'k'").set_index("time_h")["reliability"]
    gate = pd.read_csv(TABLES / "encode_dex_extremes_gate.csv").query("direction == 'A_to_B' and k_fraction == @PRIMARY_K").set_index("time_h")["z"]

    # ---- observed per time point ----
    setup, tasks, measured_changes, predicted_changes = {}, [], {}, {m: {} for m in MODELS}
    for index, t in enumerate(times):
        m_change = changes(measured, dnase[t], dnase[0])
        full_mean = (measured[dnase[t]].to_numpy().mean(axis=1) + control_mean) / 2
        evaluable = ~np.isnan(full_mean) & (full_mean >= MIN_MEAN)
        predictors = {name: changes(per_replicate[name], rna[t], rna[0]) for name in RNA_PREDICTORS}
        predictors["encode_mean_openness_prior"] = encode_prior.to_numpy()
        predictors["random_ranking"] = np.random.default_rng(42 + index).standard_normal(len(enhancers))
        predictors["control_openness_diagnostic"] = control_mean
        measured_changes[t] = pd.Series(m_change, index=enhancers)
        for model in MODELS:
            predicted_changes[model][t] = pd.Series(predictors[model], index=enhancers)
        for subset_name, subset in (("all", evaluable), ("variable", evaluable & variable.to_numpy())):
            stratum = plate_eval.strata(control_mean[subset], n_genes.to_numpy()[subset])
            ks = plate_eval.k_values(int(subset.sum()))
            setup[(t, subset_name)] = {"mask": subset, "stratum": stratum, "k": ks[PRIMARY_K]}
            for name, vector in predictors.items():
                tasks.append(((t, subset_name, name), vector[subset], m_change[subset], stratum, ks))
    results = Parallel(n_jobs=n_jobs)(delayed(observed_test)(*task) for task in tasks)
    rows = [row for result in results for row in result]
    per_time = pd.DataFrame(rows)
    per_time.to_csv(variant.tagged(TABLES / "encode_dex_eval_per_time.csv"), index=False)
    print("Observed per-time tests done", flush=True)

    # ---- events and direction (all evaluable enhancers) ----
    event_rows = []
    for t in times:
        mask = setup[(t, "all")]["mask"]
        y = measured_changes[t].to_numpy()[mask]
        for name in (*MODELS, "gene_mean_fc"):
            x = (predicted_changes[name][t].to_numpy() if name in MODELS else changes(per_replicate[name], rna[t], rna[0]))[mask]
            auc = evaluation.event_auc(x, y, EVENT_THRESHOLD)
            raw = evaluation.direction_accuracy(x, y, EVENT_THRESHOLD)
            centered = evaluation.direction_accuracy(x, y, EVENT_THRESHOLD, center=True)
            event_rows.append({"time_h": t, "predictor": name, "auroc_opening": auc["opening"]["auroc"], "auprc_opening": auc["opening"]["auprc"], "prevalence_opening": auc["opening"]["prevalence"], "n_opening": auc["opening"]["n_events"], "auroc_closing": auc["closing"]["auroc"], "auprc_closing": auc["closing"]["auprc"], "prevalence_closing": auc["closing"]["prevalence"], "n_closing": auc["closing"]["n_events"], "direction_accuracy": raw["accuracy"], "direction_accuracy_opening": raw["accuracy_opening"], "direction_accuracy_closing": raw["accuracy_closing"], "direction_accuracy_centered": centered["accuracy"]})
    events = pd.DataFrame(event_rows)
    events.to_csv(variant.tagged(TABLES / "encode_dex_eval_events.csv"), index=False)

    # ---- replicate bootstrap of the mean z and mean Spearman ----
    null_moments = per_time[per_time["k_fraction"] == PRIMARY_K].set_index(["time_h", "subset", "predictor"])[["null_mean", "null_sd"]]
    data = {
        "measured": measured.to_numpy(),
        "measured_columns": list(measured.columns),
        "replicates": {name: per_replicate[name].to_numpy() for name in RNA_PREDICTORS},
        "rna_predictors": RNA_PREDICTORS,
        "models": (*MODELS, *FROZEN),
        "replicate_columns": list(per_replicate["perd_rf"].columns),
        "fixed": {"encode_mean_openness_prior": encode_prior.to_numpy(), "control_openness_diagnostic": control_mean},
        "dnase": dnase,
        "rna": rna,
        "times": times,
        "setup": {key: {"mask": value["mask"], "k": value["k"]} for key, value in setup.items()},
        "null_moments": {key: tuple(row) for key, row in null_moments.iterrows()},
    }
    seeds = np.random.SeedSequence(42).spawn(N_BOOT)
    chunks = [seeds[i::n_jobs] for i in range(n_jobs)]
    draws = [d for chunk in Parallel(n_jobs=n_jobs)(delayed(bootstrap_chunk)(chunk, data) for chunk in chunks) for d in chunk]
    print("Bootstrap done", flush=True)
    boot = pd.DataFrame(draws)

    decision = []
    for subset_name in ("all", "variable"):
        for name in (*MODELS, *FROZEN, "gene_mean_fc", *FIXED_PREDICTORS):
            for metric in ("z", "spearman"):
                cols = [c for c in boot.columns if c[1] == subset_name and c[2] == name and c[3] == metric]
                if not cols:
                    continue
                observed_rows = per_time[(per_time["subset"] == subset_name) & (per_time["predictor"] == name) & (per_time["k_fraction"] == PRIMARY_K)]
                observed = float(observed_rows["z" if metric == "z" else "spearman"].mean())
                draws_mean = boot[cols].mean(axis=1).to_numpy()
                level = 0.975 if name in (*MODELS, *FROZEN) else 0.95
                tail = (1 - level) / 2 * 100
                low, high = np.nanpercentile(draws_mean, [tail, 100 - tail])
                se = float(np.nanstd(draws_mean, ddof=1))
                decision.append({"subset": subset_name, "predictor": name, "metric": f"mean_{metric}_over_11_times", "observed": observed, "ci_low": float(low), "ci_high": float(high), "level": level, "bootstrap_se": se, "min_detectable_effect": 2.8 * se})
        # Amendment 07 P1: paired difference (variant - frozen) on the same replicate draws.
        for model in ("rf", "ridge") if FROZEN else ():
            for metric in ("z", "spearman"):
                cols_v = [c for c in boot.columns if c[1] == subset_name and c[2] == f"perd_{model}" and c[3] == metric]
                cols_f = [c for c in boot.columns if c[1] == subset_name and c[2] == f"frozen_{model}" and c[3] == metric]
                obs = per_time[(per_time["subset"] == subset_name) & (per_time["k_fraction"] == PRIMARY_K)].set_index(["predictor", "time_h"])["z" if metric == "z" else "spearman"]
                observed = float(obs.loc[f"perd_{model}"].mean() - obs.loc[f"frozen_{model}"].mean())
                diff = boot[cols_v].mean(axis=1).to_numpy() - boot[cols_f].mean(axis=1).to_numpy()
                low, high = np.nanpercentile(diff, [1.25, 98.75])
                decision.append({"subset": subset_name, "predictor": f"diff_{model}_{variant.tag()}_minus_frozen", "metric": f"mean_{metric}_over_11_times", "observed": observed, "ci_low": float(low), "ci_high": float(high), "level": 0.975, "bootstrap_se": float(np.nanstd(diff, ddof=1)), "min_detectable_effect": 2.8 * float(np.nanstd(diff, ddof=1))})
    decision = pd.DataFrame(decision)
    decision.to_csv(variant.tagged(TABLES / "encode_dex_eval_decision.csv"), index=False)

    # ---- ceiling and lag matrix ----
    primary = per_time[(per_time["subset"] == "all") & (per_time["k_fraction"] == PRIMARY_K)]
    ceiling = []
    for model in MODELS:
        for t in times:
            row = primary[(primary["predictor"] == model) & (primary["time_h"] == t)].iloc[0]
            rel = reliability.get(t, np.nan)
            ceiling.append({"predictor": model, "time_h": t, "spearman": row["spearman"], "reliability": rel, "spearman_over_sqrt_reliability": row["spearman"] / np.sqrt(rel) if rel > 0 else np.nan, "z": row["z"], "split_half_gate_z": gate.get(t, np.nan), "z_over_gate_z": row["z"] / gate.get(t, np.nan)})
    ceiling = pd.DataFrame(ceiling)
    ceiling.to_csv(variant.tagged(TABLES / "encode_dex_eval_ceiling.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lag_tables = {}
    for ax, model in zip(axes, MODELS):
        lag = np.zeros((len(times), len(times)))
        for i, ti in enumerate(times):
            for j, tj in enumerate(times):
                mask = setup[(ti, "all")]["mask"] & setup[(tj, "all")]["mask"]
                lag[i, j] = evaluation.correlation(predicted_changes[model][ti].to_numpy()[mask], measured_changes[tj].to_numpy()[mask])
        lag_tables[model] = pd.DataFrame(lag, index=[f"RNA {t:g}h" for t in times], columns=[f"DNase {t:g}h" for t in times])
        limit = np.nanmax(np.abs(lag))
        image = ax.imshow(lag, cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_xticks(range(len(times)))
        ax.set_xticklabels([f"{t:g}" for t in times])
        ax.set_yticks(range(len(times)))
        ax.set_yticklabels([f"{t:g}" for t in times])
        ax.set_xlabel("DNase time (h)")
        ax.set_ylabel("RNA time (h)")
        ax.set_title(f"{model}: Spearman, predicted vs measured change")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(variant.tagged(FIGURES / "encode_dex_lag_matrix.png"), dpi=150)
    plt.close(fig)
    pd.concat(lag_tables, names=["predictor"]).to_csv(variant.tagged(TABLES / "encode_dex_lag_matrix.csv"))

    # ---- per-time figure (primary z by time, models and baselines) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in (*MODELS, "gene_mean_fc", "encode_mean_openness_prior", "random_ranking"):
        rows_name = primary[primary["predictor"] == name]
        ax.plot(rows_name["time_h"], rows_name["z"], marker="o", label=name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(3, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Time after dexamethasone (h)")
    ax.set_ylabel("z, extreme-prediction test (K = 1%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(variant.tagged(FIGURES / "encode_dex_z_by_time.png"), dpi=150)
    plt.close(fig)

    summary = {"decision": decision.to_dict("records"), "per_time_primary": primary[["time_h", "predictor", "K", "S", "z", "z_opening", "z_closing", "p_one_sided", "spearman", "pearson", "n_evaluable"]].to_dict("records")}
    variant.tagged(TABLES / "encode_dex_eval_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(decision.round(3).to_string(index=False))
        print(primary[primary["predictor"].isin(MODELS)][["time_h", "predictor", "z", "z_opening", "z_closing", "spearman"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
