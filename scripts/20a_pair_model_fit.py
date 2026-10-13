"""Amendment 10 E4: perturbation-pair model, leave-one-perturbation-out (fit + predict). Data prep is PERD-blind.

Usage: python scripts/20a_pair_model_fit.py --models E4,B0,noBase --units "A549 dexamethasone",... (default: all units)
Writes data/processed/pair_model/pred_<model>_<unit>.parquet (enhancers x held-out conditions, predicted change) and
data/processed/pair_model/conditions.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

from perd_perturbation import pair_model
from perd_perturbation.pair_model import Condition

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
OUT = PROC / "pair_model"
N_TRAIN, MIN_MEAN, MIN_KB = 20_000, 1.0, 0.2
DATASETS = {"encode_dex": "a549_dex", "encode_k562": "k562", "encode_tcell": "tcell", "encode_hct116": "hct116", "encode_sknsh": "sknsh"}


def build_conditions() -> list[Condition]:
    out = []
    files = pd.DataFrame(json.loads((ROOT / "data" / "encode_dex_manifest.json").read_text())["files"])
    dn = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "DNase"].groupby("time_h")}
    rn = {t: g["file_accession"].tolist() for t, g in files[files["assay"] == "RNA"].groupby("time_h")}
    for t in sorted(x for x in dn if x != 0.0):
        out.append(Condition("encode_dex", "A549 dexamethasone", f"A549 dex {t:g}h", dn[0.0], dn[t], rn[0.0], rn[t], 1.1292896530999152, {"time_h": t}))
    gates = json.loads((TABLES / "a06_gates.json").read_text())
    noise = {float(k): v["event_threshold_p99"] for k, v in gates["k562"]["dmso_noise_by_time"].items()}
    s = pd.read_csv(PROC / "encode_k562" / "samples.csv")
    acc, rna = s[s["assay"] == "ATAC-seq"], s[s["assay"] != "ATAC-seq"]
    gate = pd.read_csv(TABLES / "a06_k562_gate.csv")
    for r in gate[gate["replicable"]].sort_values(["drug", "time_h"]).itertuples():
        pick = lambda d, drug: d[(d["drug"] == drug) & (d["time_h"] == r.time_h)]["file_accession"].tolist()  # noqa: E731
        out.append(Condition("encode_k562", f"K562 {r.drug}", f"K562 {r.drug} {r.time_h:g}h", pick(acc, "DMSO"), pick(acc, r.drug), pick(rna, "DMSO"), pick(rna, r.drug), noise[float(r.time_h)], {"group": r.group, "time_h": r.time_h}))
    s = pd.read_csv(PROC / "encode_tcell" / "samples.csv")
    for pair, g in s.groupby("pair"):
        a, r_ = g[g["assay"] == "DNase-seq"], g[g["assay"] != "DNase-seq"]
        sel = lambda d, st: d[d["state"] == st]["file_accession"].tolist()  # noqa: E731
        out.append(Condition("encode_tcell", "T-cell activation", f"T {pair}", sel(a, "resting"), sel(a, "activated"), sel(r_, "resting"), sel(r_, "activated"), gates["tcell"]["event_threshold"]))
    g9 = pd.read_csv(TABLES / "a09_gate_per_unit.csv")
    thr = json.loads((TABLES / "a09_gates.json").read_text())["hct116"]["event_threshold"]
    for dataset, units in (("encode_hct116", g9[(g9["dataset"] == "hct116") & g9["replicable"]]["unit"].tolist()), ("encode_sknsh", ["retinoic acid"])):
        s = pd.read_csv(PROC / dataset / "samples.csv")
        for unit in units:
            g = s[s["unit"] == unit]
            a, r_ = g[g["assay"] == "DNase-seq"], g[g["assay"] != "DNase-seq"]
            sel = lambda d, st: d[d["state"] == st]["file_accession"].tolist()  # noqa: E731
            label = f"HCT116 {unit}" if dataset == "encode_hct116" else "SK-N-SH retinoic acid"
            out.append(Condition(dataset, label, label, sel(a, "untreated"), sel(a, "treated"), sel(r_, "untreated"), sel(r_, "treated"), thr if dataset == "encode_hct116" else None))
    return out


def enhancer_tables() -> pair_model.EnhancerTables:
    training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    enh = pd.Index(training["enhancer_id"])
    rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    genes = rna.index
    gh = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id").loc[enh]
    length = ((gh["end"] - gh["start"]) / 1000).clip(lower=MIN_KB)
    activity = pd.read_parquet(PROC / "tf_activity" / "activity_encode167.parquet")
    mapping = pd.read_csv(TABLES / "motif_archetype_tfs.csv")
    arch_tfs = {r.archetype: [t for t in str(r.tfs).split(";") if t in activity.index] for r in mapping.itertuples()}
    arch_tfs = {a: v for a, v in arch_tfs.items() if v}
    counts = pd.read_parquet(PROC / "motifs" / "enhancer_archetypes.parquet")
    counts = counts[counts["archetype"].isin(arch_tfs) & counts["enhancer_id"].isin(enh)]
    counts["density"] = counts["n_matches"] / counts["enhancer_id"].map(length).to_numpy()
    dens = counts.pivot_table(index="enhancer_id", columns="archetype", values="density", fill_value=0.0).reindex(index=enh, columns=sorted(arch_tfs), fill_value=0.0)
    arch_z = (dens - dens.mean()) / dens.std(ddof=0).replace(0, 1.0)
    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"].reindex(enh)
    scored = sorted({t for v in binding.dropna() for t in v} & set(activity.index))
    bound = pd.DataFrame(0.0, index=enh, columns=scored)
    for e, tfs in binding.dropna().items():
        hit = [t for t in tfs if t in bound.columns]
        bound.loc[e, hit] = 1.0
    gpos = {g: i for i, g in enumerate(genes)}
    rows, cols = [], []
    for i, tg in enumerate(gh["target_genes"]):
        for g in (tg if tg is not None else []):
            if g in gpos:
                rows.append(i)
                cols.append(gpos[g])
    targets = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(enh), len(genes)))
    targets.sum_duplicates()
    targets.data[:] = 1.0
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet").loc[enh]
    keep = [c for c in openness.columns if not c.startswith("SK-N-SH_treated")]
    static = pd.DataFrame({"prior_mean": openness[keep].mean(axis=1), "prior_sd": openness[keep].std(axis=1), "length_kb": length,
                           "n_targets": np.diff(targets.indptr), "n_bound": bound.sum(axis=1)}, index=enh)
    return pair_model.EnhancerTables(enh, arch_z, bound, targets, genes, static, {a: arch_tfs[a] for a in arch_z.columns})


def load_dataset(dataset: str) -> dict:
    out = PROC / dataset
    return {"open": pd.read_parquet(out / "enhancer_openness_replicates.parquet"), "rna": pd.read_parquet(out / "rna_log2fpkm_replicates.parquet"),
            "act": pd.read_parquet(PROC / "tf_activity" / f"activity_{DATASETS[dataset]}.parquet")}


def condition_matrix(t, data, cond) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features, measured change, evaluable mask) for one condition over all enhancers."""
    o = data[cond.dataset]["open"].reindex(t.enhancers)
    control, treated = o[cond.acc_control].mean(axis=1), o[cond.acc_treated].mean(axis=1)
    rna = data[cond.dataset]["rna"]
    d_rna = rna[cond.rna_treated].mean(axis=1) - rna[cond.rna_control].mean(axis=1)
    x = pair_model.condition_features(t, pair_model.delta_activity(data[cond.dataset]["act"], cond), d_rna, control)
    evaluable = (control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)).to_numpy()
    return x, (treated - control).to_numpy(), evaluable


def columns_for(model: str, names: list[str]) -> np.ndarray:
    if model == "B0":
        keep = ["baseline", *pair_model.STATIC]
    elif model == "noBase":
        keep = [n for n in names if n != "baseline"]
    else:
        keep = names
    return np.array([names.index(n) for n in keep])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="E4,B0,noBase")
    parser.add_argument("--units", default="")
    args = parser.parse_args()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    conditions = build_conditions()
    units = sorted({c.unit for c in conditions})
    (OUT / "conditions.json").write_text(json.dumps([c.__dict__ for c in conditions], indent=1, default=str) + "\n")
    t = enhancer_tables()
    names = t.feature_names()
    data = {d: load_dataset(d) for d in DATASETS}
    weights = pair_model.unit_weights(conditions)
    print(f"{len(conditions)} conditions, {len(units)} units, {len(names)} features; tables {time.time() - start:.0f}s", flush=True)

    # Fixed training rows: 20,000 evaluable enhancers per condition (seed 42, condition order), chosen without the change.
    rng = np.random.default_rng(42)
    train_x, train_y, train_w, train_cond = [], [], [], []
    for cond in conditions:
        x, y, ev = condition_matrix(t, data, cond)
        idx = np.flatnonzero(ev)
        pick = np.sort(rng.choice(idx, size=min(N_TRAIN, len(idx)), replace=False))
        train_x.append(x[pick])
        train_y.append(y[pick])
        train_w.append(np.full(len(pick), weights[cond.name] / len(pick), dtype=np.float64))
        train_cond.append(np.full(len(pick), cond.unit, dtype=object))
    train_x, train_y, train_w, train_cond = np.vstack(train_x), np.concatenate(train_y), np.concatenate(train_w), np.concatenate(train_cond)
    print(f"training rows {len(train_y):,}; {time.time() - start:.0f}s", flush=True)

    selected = [u for u in units if not args.units or u in args.units.split(",")]
    for unit in selected:
        mask = train_cond != unit
        held = [c for c in conditions if c.unit == unit]
        held_x = {c.name: condition_matrix(t, data, c)[0] for c in held}
        for model in args.models.split(","):
            t0 = time.time()
            cols = columns_for(model, names)
            rf = RandomForestRegressor(n_estimators=500, max_features=max(1, len(cols) // 3), min_samples_leaf=100, random_state=42, n_jobs=n_jobs)
            rf.fit(train_x[mask][:, cols], train_y[mask], sample_weight=train_w[mask])
            pred = pd.DataFrame({c.name: rf.predict(held_x[c.name][:, cols]) for c in held}, index=t.enhancers)
            pred.to_parquet(OUT / f"pred_{model}_{unit.replace(' ', '_')}.parquet")
            if model == "E4":
                imp = pd.Series(rf.feature_importances_, index=[names[i] for i in cols]).sort_values(ascending=False)
                imp.head(30).to_csv(OUT / f"impurity_importance_E4_{unit.replace(' ', '_')}.csv", header=["importance"])
            print(f"{unit} | {model}: {mask.sum():,} train rows, {len(held)} held-out conditions, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"total {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
