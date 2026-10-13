"""Amendment 05 B, C, D (secondary): extreme test within candidate / PERD-skilled / GR-bound subsets with the
ridge - RF paired difference; BIRD-style module-level test; glucocorticoid-receptor site enrichment.
Definitions fixed in execution.md (amendment 05) before running.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

from perd_perturbation import atac, evaluation, plate_eval
from perd_perturbation import variant


ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
TABLES = ROOT / "results" / "tables"
GR_DIR = ROOT / "data" / "raw" / "encode_gr"
MIN_MEAN = 1.0
N_BOOT = 2000
MODELS = ("perd_rf", "perd_ridge")
N_MODULES = 1000
MIN_MODULE_MEMBERS = 5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


section_e = load_module("section_e", ROOT / "scripts" / "09f_encode_dex_evaluation.py")
pipeline = load_module("pipeline", ROOT / "scripts" / "11b_perd_pipeline_dex.py")


def gr_masks(enhancers: pd.Index) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """GR-bound: enhancer (1-based closed -> 0-based half-open) overlaps a conservative IDR peak."""
    genehancer = pd.read_parquet(PERD / "genehancer_enhancers.parquet").set_index("enhancer_id").loc[enhancers]
    starts, ends, chroms = genehancer["start"].to_numpy() - 1, genehancer["end"].to_numpy(), genehancer["chrom"].to_numpy()
    manifest = json.loads((ROOT / "data" / "encode_gr_manifest.json").read_text())["files"]
    per_time, union = {}, np.zeros(len(enhancers), dtype=bool)
    for record in manifest:
        peaks = pd.read_csv(GR_DIR / f"{record['file_accession']}.bed.gz", sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
        merged = atac.merge_intervals(peaks)
        hit = np.zeros(len(enhancers), dtype=bool)
        for chrom, (ms, me) in merged.items():
            m = chroms == chrom
            if m.any():
                hit[m] = atac.overlaps_merged(starts[m], ends[m], ms, me)
        per_time[float(record["time_h"])] = hit
        union |= hit
    return union, per_time


def skilled_mask(enhancers: pd.Index) -> np.ndarray:
    held_out = pd.read_parquet(PERD / "netopen_rf_predictions.parquet").set_index("enhancer_id")
    r = {e: float(np.corrcoef(np.asarray(y), np.asarray(p))[0, 1]) for e, y, p in zip(held_out.index, held_out["y_test"], held_out["y_pred"])}
    return pd.Series(r).reindex(enhancers).fillna(-1).to_numpy() > 0.5


def modules_for(enhancers: pd.Index) -> np.ndarray:
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet").loc[enhancers]
    values = openness.to_numpy()
    sd = values.std(axis=1, keepdims=True)
    z = (values - values.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1)
    return MiniBatchKMeans(n_clusters=N_MODULES, random_state=42, n_init=10, batch_size=4096).fit_predict(z)


def observed_extreme(key, vector, measured, stratum, k):
    result = plate_eval.extreme_test_components(vector, measured, stratum, {0.01: k} if k > 0 else {}, n_perm=2000, seed=42)
    return key, result.get(0.01)


def bootstrap_chunk(seed_sequences, data):
    m_col = {a: i for i, a in enumerate(data["measured_columns"])}
    r_col = {a: i for i, a in enumerate(data["replicate_columns"])}
    out = []
    for seed_sequence in seed_sequences:
        rng = np.random.default_rng(seed_sequence)

        def pick(accessions):
            return [accessions[i] for i in rng.integers(0, len(accessions), len(accessions))]

        m_ctrl = data["measured"][:, [m_col[a] for a in pick(data["dnase"][0])]].mean(axis=1)
        rna_ctrl = pick(data["rna"][0])
        p_ctrl = {n: data["replicates"][n][:, [r_col[a] for a in rna_ctrl]].mean(axis=1) for n in data["replicates"]}
        z = {}
        for t in data["times"]:
            m = data["measured"][:, [m_col[a] for a in pick(data["dnase"][t])]].mean(axis=1) - m_ctrl
            rna_t = pick(data["rna"][t])
            p = {n: data["replicates"][n][:, [r_col[a] for a in rna_t]].mean(axis=1) - p_ctrl[n] for n in data["replicates"]}
            for subset, s in data["subsets"][t].items():
                for name in MODELS:
                    if (subset, t, name) not in data["null"]:
                        continue
                    stat, _, _ = plate_eval.extreme_statistics(p[name][s["mask"]], m[s["mask"]], s["k"])
                    mean_null, sd_null = data["null"][(subset, t, name)]
                    z[(subset, t, name, "z")] = (stat - mean_null) / sd_null
            mod = data["modules"][t]
            mm = mod["matrix"] @ m
            for name in (*MODELS, "gene_mean_fc"):
                pm = mod["matrix"] @ p[name]
                z[("modules", t, name, "spearman")] = evaluation.correlation(pm, mm)
                if ("modules", t, name) in data["null"]:
                    stat, _, _ = plate_eval.extreme_statistics(pm, mm, mod["k"])
                    mean_null, sd_null = data["null"][("modules", t, name)]
                    z[("modules", t, name, "z")] = (stat - mean_null) / sd_null
        out.append(z)
    return out


def main() -> None:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    dnase, rna, measured, per_replicate, n_genes, encode_prior, _variable = section_e.load()
    enhancers = measured.index
    times = sorted(t for t in dnase if t != 0)
    control_mean = measured[dnase[0]].mean(axis=1).to_numpy()
    rna_symbols = set(pd.read_parquet(ROOT / "data" / "processed" / "encode_dex" / "rna_log2fpkm_replicates.parquet").index)
    candidate, counts = pipeline.candidate_sets(enhancers, rna_symbols)
    gr_union, gr_time = gr_masks(enhancers)
    masks = {name: enhancers.isin(idx) for name, idx in candidate.items()}
    masks["S_skilled"] = skilled_mask(enhancers)
    masks["GR_bound"] = gr_union
    masks["all"] = np.ones(len(enhancers), dtype=bool)
    summary = {"subset_sizes_fitted": {k: int(v.sum()) for k, v in masks.items()}, "gr_bound_by_time": {str(k): int(v.sum()) for k, v in gr_time.items()}}
    print(json.dumps(summary, indent=2), flush=True)

    module_labels = modules_for(enhancers)
    tasks, data_subsets, data_modules, gr_rows = [], {}, {}, []
    for t in times:
        m_change = section_e.changes(measured, dnase[t], dnase[0])
        full_mean = (measured[dnase[t]].to_numpy().mean(axis=1) + control_mean) / 2
        evaluable = ~np.isnan(full_mean) & (full_mean >= MIN_MEAN)
        predicted = {n: section_e.changes(per_replicate[n], rna[t], rna[0]) for n in (*MODELS, "gene_mean_fc")}
        data_subsets[t] = {}
        for subset, mask in masks.items():
            sel = evaluable & mask
            if sel.sum() < 1000:
                continue
            stratum = plate_eval.strata(control_mean[sel], n_genes.to_numpy()[sel])
            k = plate_eval.k_values(int(sel.sum()))[0.01]
            data_subsets[t][subset] = {"mask": sel, "k": k}
            for name in MODELS:
                tasks.append(((subset, t, name), predicted[name][sel], m_change[sel], stratum, k))
        # Modules: mean over evaluable members.
        member = evaluable.copy()
        labels = module_labels[member]
        counts_per_module = np.bincount(labels, minlength=N_MODULES)
        keep = np.flatnonzero(counts_per_module >= MIN_MODULE_MEMBERS)
        rows = np.searchsorted(keep, labels)
        valid = np.isin(labels, keep)
        idx = np.flatnonzero(member)[valid]
        matrix = sparse.csr_matrix((1.0 / counts_per_module[labels[valid]], (rows[valid], idx)), shape=(len(keep), len(enhancers)))
        module_control = matrix @ np.nan_to_num(control_mean)
        module_stratum = pd.qcut(pd.Series(module_control).rank(method="first"), 10, labels=False).to_numpy()
        k_modules = max(1, round(0.05 * len(keep)))
        data_modules[t] = {"matrix": matrix, "k": k_modules}
        mm = matrix @ np.nan_to_num(m_change)
        for name in (*MODELS, "gene_mean_fc"):
            tasks.append((("modules", t, name), matrix @ predicted[name], mm, module_stratum, k_modules))
        # D: GR enrichment among top 1% predicted and measured openers.
        stratum = pd.qcut(pd.Series(control_mean[evaluable]).rank(method="first"), 10, labels=False).to_numpy()
        for label, score in (("perd_rf top1%", predicted["perd_rf"]), ("perd_ridge top1%", predicted["perd_ridge"]), ("measured top1%", m_change)):
            s = score[evaluable]
            top = s >= np.quantile(s, 0.99)
            for gr_name, gr_mask in (("union", gr_union), ("time_matched", gr_time[float(t)])):
                cmh = evaluation.cmh_odds_ratio(top, gr_mask[evaluable], stratum)
                gr_rows.append({"time_h": t, "selection": label, "gr_definition": gr_name, "fraction_gr_in_selection": float(gr_mask[evaluable][top].mean()), "fraction_gr_all_evaluable": float(gr_mask[evaluable].mean()), **cmh})
    gr_table = pd.DataFrame(gr_rows)
    gr_table.to_csv(variant.tagged(TABLES / "a05_gr_enrichment.csv"), index=False)

    results = Parallel(n_jobs=n_jobs)(delayed(observed_extreme)(*task) for task in tasks)
    observed = pd.DataFrame([{"subset": k[0], "time_h": k[1], "predictor": k[2], **r} for k, r in results if r is not None])
    module_spearman = []
    for t in times:
        m_change = section_e.changes(measured, dnase[t], dnase[0])
        mm = data_modules[t]["matrix"] @ np.nan_to_num(m_change)
        for name in (*MODELS, "gene_mean_fc"):
            pm = data_modules[t]["matrix"] @ section_e.changes(per_replicate[name], rna[t], rna[0])
            module_spearman.append({"time_h": t, "predictor": name, "module_spearman": evaluation.correlation(pm, mm), "n_modules": int(data_modules[t]["matrix"].shape[0])})
    module_spearman = pd.DataFrame(module_spearman)
    observed.to_csv(variant.tagged(TABLES / "a05_subsets_modules_per_time.csv"), index=False)
    module_spearman.to_csv(variant.tagged(TABLES / "a05_modules_spearman_per_time.csv"), index=False)
    print("Observed tests done", flush=True)

    data = {
        "measured": measured.to_numpy(),
        "measured_columns": list(measured.columns),
        "replicates": {n: np.nan_to_num(per_replicate[n].to_numpy()) for n in (*MODELS, "gene_mean_fc")},
        "replicate_columns": list(per_replicate["perd_rf"].columns),
        "dnase": dnase,
        "rna": rna,
        "times": times,
        "subsets": data_subsets,
        "modules": data_modules,
        "null": {(r.subset, r.time_h, r.predictor): (r.null_mean, r.null_sd) for r in observed.itertuples()},
    }
    data["measured"] = np.nan_to_num(data["measured"])
    seeds = np.random.SeedSequence(42).spawn(N_BOOT)
    chunks = [seeds[i::n_jobs] for i in range(n_jobs)]
    boot = pd.DataFrame([d for chunk in Parallel(n_jobs=n_jobs)(delayed(bootstrap_chunk)(c, data) for c in chunks) for d in chunk])
    print("Bootstrap done", flush=True)

    decision = []
    subsets = sorted({c[0] for c in boot.columns})
    for subset in subsets:
        for metric in ("z", "spearman"):
            per_model = {}
            for name in (*MODELS, "gene_mean_fc"):
                cols = [c for c in boot.columns if c[0] == subset and c[2] == name and c[3] == metric]
                if not cols:
                    continue
                draws = boot[cols].mean(axis=1).to_numpy()
                per_model[name] = draws
                if metric == "z":
                    obs = float(observed[(observed["subset"] == subset) & (observed["predictor"] == name)]["z"].mean())
                else:
                    obs = float(module_spearman[module_spearman["predictor"] == name]["module_spearman"].mean())
                level = 0.975 if name in MODELS else 0.95
                tail = (1 - level) / 2 * 100
                low, high = np.nanpercentile(draws, [tail, 100 - tail])
                decision.append({"subset": subset, "metric": f"mean_{metric}", "predictor": name, "observed": obs, "ci_low": float(low), "ci_high": float(high), "level": level, "bootstrap_se": float(np.nanstd(draws, ddof=1))})
            if "perd_ridge" in per_model and "perd_rf" in per_model:
                diff = per_model["perd_ridge"] - per_model["perd_rf"]
                low, high = np.nanpercentile(diff, [2.5, 97.5])
                obs_diff = [d for d in decision if d["subset"] == subset and d["metric"] == f"mean_{metric}"]
                observed_diff = next(d["observed"] for d in obs_diff if d["predictor"] == "perd_ridge") - next(d["observed"] for d in obs_diff if d["predictor"] == "perd_rf")
                decision.append({"subset": subset, "metric": f"mean_{metric}", "predictor": "ridge_minus_rf", "observed": observed_diff, "ci_low": float(low), "ci_high": float(high), "level": 0.95, "bootstrap_se": float(np.nanstd(diff, ddof=1))})
    decision = pd.DataFrame(decision)
    decision.to_csv(variant.tagged(TABLES / "a05_subsets_modules_decision.csv"), index=False)
    variant.tagged(TABLES / "a05_subsets_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    gr_summary = gr_table.groupby(["selection", "gr_definition"])[["fraction_gr_in_selection", "fraction_gr_all_evaluable", "odds_ratio"]].mean()
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(decision.round(3).to_string(index=False))
        print(gr_summary.round(3).to_string())
        print(module_spearman.pivot(index="time_h", columns="predictor", values="module_spearman").round(3).to_string())


if __name__ == "__main__":
    main()
