"""End-to-end demo on the toy data: the four questions of the report, at small scale.

It runs the same library code as the full analysis (PERD-RF per enhancer, TF activity by ULM, motif inputs, the
pair model) on 150 enhancers and three perturbation series, and writes small tables to demo/output/.

  Q1  How well does PERD-RF predict openness in tissues it never saw? (5-fold, grouped by tissue type)
  Q2  Does it predict the openness change a perturbation causes? (NSPE before, after, change)
  Q3  Do TF activity and motif inputs help? (unseen tissues, and the perturbation change)
  Q4  Does a model trained on real before/after pairs help? (leave one perturbation out, with a baseline-only control)

Usage: pixi run demo  [--n-jobs 4]      (about 12 minutes on 4 CPUs)

The numbers will not match the report: 150 enhancers instead of 120,100, fewer perturbations, and TF activity scored
on a reduced gene set. The point is to show the method running end to end on real data.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

from perd_perturbation import pair_model, tf_activity
from perd_perturbation.perd_rf import run_perd_model

ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "demo" / "toy_data"
MIN_MEAN, Z_MIN, TOP_N, MIN_KB, SEED = 1.0, 2.0, 20, 0.2, 42
ARCH = "act:arch:"
# The full pair model uses min_samples_leaf = 100 on 800,000 rows; the toy table has about 1,600 rows.
PAIR_LEAF = 5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


split = load_module("split", ROOT / "scripts" / "11d_tissue_grouped_split.py")       # tissue folds, accuracy metrics
part_m = load_module("part_m", ROOT / "scripts" / "19a_motif_activity_heldout.py")   # archetype activity
shuffle = load_module("shuffle", ROOT / "scripts" / "17a_shuffled_regulon_control.py")  # shuffled TF -> target lists


def nspe(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Normalized squared prediction error: 0 = perfect, 1 = no better than predicting the mean."""
    ok = ~np.isnan(observed) & ~np.isnan(predicted)
    o, p = observed[ok], predicted[ok]
    return float(np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2))


def load() -> dict:
    with gzip.open(TOY / "collectri.tsv.gz", "rt") as handle:
        network_raw = pd.read_csv(handle, sep="\t")
    return {
        "enhancers": pd.read_parquet(TOY / "enhancers.parquet"),
        "rna": pd.read_parquet(TOY / "encode_rna.parquet").astype(float),
        "openness": pd.read_parquet(TOY / "encode_openness.parquet").astype(float),
        "network": tf_activity.network_from_collectri(network_raw),
        "motifs": pd.read_parquet(TOY / "motif_counts.parquet"),
        "motif_scale": pd.read_csv(TOY / "motif_density_scale.csv").set_index("archetype"),
        "pert_open": pd.read_parquet(TOY / "perturbation_openness.parquet").astype(float),
        "pert_rna": pd.read_parquet(TOY / "perturbation_rna.parquet").astype(float),
        "conditions": [pair_model.Condition(**{k: c[k] for k in ("dataset", "unit", "name")}, acc_control=c["acc_control"], acc_treated=c["acc_treated"],
                                            rna_control=c["rna_control"], rna_treated=c["rna_treated"])
                       for c in json.loads((TOY / "conditions.json").read_text())],
    }


def activities(d: dict) -> dict:
    """TF activity (ULM t-values) for the ENCODE cells and every perturbation sample, plus a shuffled-target control."""
    mean, sd = tf_activity.reference_scale(d["rna"])
    weights = tf_activity.weight_matrix(d["network"], mean.index)
    encode = tf_activity.activity(d["rna"], mean, sd, weights)
    pert = tf_activity.activity(d["pert_rna"], mean, sd, weights)  # genes missing from a sample count as z = 0
    null = shuffle.shuffled_network(d["network"], mean.index, encode.index, seed=SEED)
    null_w = tf_activity.weight_matrix(null, mean.index)
    return {"encode": encode, "pert": pert, "encode_shuf": tf_activity.activity(d["rna"], mean, sd, null_w),
            "pert_shuf": tf_activity.activity(d["pert_rna"], mean, sd, null_w)}


def motif_sets(d: dict, archetypes: set[str]) -> dict[str, list[str]]:
    """Per enhancer: archetypes (with activity) whose density z >= 2, at most the 20 highest (as in the full analysis)."""
    m = d["motifs"][d["motifs"]["archetype"].isin(archetypes) & (d["motifs"]["density_z"] >= Z_MIN)]
    top = m.sort_values("density_z", ascending=False).groupby("enhancer_id").head(TOP_N)
    return top.groupby("enhancer_id")["archetype"].apply(list).to_dict()


def variants(d: dict, act: dict) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Input versions of PERD-RF: name -> (feature lists, ENCODE feature table, perturbation feature table)."""
    enh = d["enhancers"]
    base = enh[["enhancer_id", "usable_genes"]].copy()
    binding = enh.set_index("enhancer_id")["tf_genes"]
    targets = enh.set_index("enhancer_id")["target_genes"]
    arch_enc = part_m.archetype_activity(act["encode"])
    arch_pert = part_m.archetype_activity(act["pert"])
    sets = motif_sets(d, set(arch_enc.index))
    add_arch = lambda lists: lists.assign(usable_genes=[list(g) + [ARCH + a for a in sets.get(e, [])] for e, g in zip(lists["enhancer_id"], lists["usable_genes"])])  # noqa: E731
    arch_rows = lambda table, cols: table.rename(index=lambda a: ARCH + a)[cols]  # noqa: E731
    rna, prna = d["rna"], d["pert_rna"]
    with_act = lambda r, a: tf_activity.with_activity(r, a)  # noqa: E731
    tfadd = tf_activity.feature_lists(base, binding, targets, set(act["encode"].index), "tfadd")[["enhancer_id", "usable_genes"]]
    return {
        "PERD-RF (RNA only)": (base, rna, prna),
        "+ shuffled activity (control)": (tfadd, with_act(rna, act["encode_shuf"]), with_act(prna, act["pert_shuf"])),
        "+ TF activity": (tfadd, with_act(rna, act["encode"]), with_act(prna, act["pert"])),
        "+ motifs": (add_arch(base), pd.concat([rna, arch_rows(arch_enc, rna.columns)]), pd.concat([prna, arch_rows(arch_pert, prna.columns)])),
        "+ TF activity + motifs": (add_arch(tfadd), pd.concat([with_act(rna, act["encode"]), arch_rows(arch_enc, rna.columns)]),
                                   pd.concat([with_act(prna, act["pert"]), arch_rows(arch_pert, prna.columns)])),
    }


def held_out_tissues(lists, features, openness, n_jobs) -> np.ndarray:
    """Out-of-fold predictions (enhancers x cells), each cell predicted by models that never saw its tissue type."""
    cells = list(openness.columns)
    folds = split.fold_assignment(cells)
    ids = lists["enhancer_id"].tolist()
    predicted = np.full((len(ids), len(cells)), np.nan)
    for fold in range(split.N_FOLDS):
        train = [c for c, f in zip(cells, folds) if f != fold]
        test = [c for c, f in zip(cells, folds) if f == fold]
        table = run_perd_model(lists, features[train], openness[train], features[test], model="rf", n_jobs=n_jobs)
        predicted[:, [cells.index(c) for c in test]] = table.loc[ids, test].to_numpy()
    return predicted


def q1_q3_unseen_tissues(d, versions, n_jobs) -> pd.DataFrame:
    actual = d["openness"].loc[d["enhancers"]["enhancer_id"]].to_numpy()
    rows = []
    # Paper-style random split (110 train / 57 test cells) for frozen PERD-RF, for comparison.
    cells = list(d["openness"].columns)
    test = list(np.random.default_rng(SEED).choice(cells, 57, replace=False))
    train = [c for c in cells if c not in test]
    lists, feats, _ = versions["PERD-RF (RNA only)"]
    table = run_perd_model(lists, feats[train], d["openness"][train], feats[test], model="rf", n_jobs=n_jobs)
    a, p = d["openness"].loc[lists["enhancer_id"], test].to_numpy(), table.loc[lists["enhancer_id"], test].to_numpy()
    rows.append({"version": "PERD-RF (RNA only)", "scheme": "random 110/57 split", **split.metrics(a, p),
                 "nspe_per_cell_median": float(np.median([nspe(a[:, j], p[:, j]) for j in range(a.shape[1])]))})
    for name, (lists, feats, _) in versions.items():
        p = held_out_tissues(lists, feats, d["openness"], n_jobs)
        rows.append({"version": name, "scheme": "tissue held out", **split.metrics(actual, p),
                     "nspe_per_cell_median": float(np.median([nspe(actual[:, j], p[:, j]) for j in range(actual.shape[1])]))})
        print(f"  unseen tissues | {name}: per-cell NSPE {rows[-1]['nspe_per_cell_median']:.3f}", flush=True)
    return pd.DataFrame(rows)


def q2_q3_perturbations(d, versions, n_jobs) -> tuple[pd.DataFrame, dict]:
    """Fit each version on the 167 ENCODE cells, predict every perturbation RNA sample, score before / after / change."""
    rows, frozen_change = [], {}
    for name, (lists, feats, pfeats) in versions.items():
        if name.startswith("+ shuffled"):
            continue
        pred = run_perd_model(lists, feats, d["openness"], pfeats.fillna(0.0), model="rf", n_jobs=n_jobs)
        for c in d["conditions"]:
            o = d["pert_open"].reindex(pred.index)
            mc, mt = o[c.acc_control].mean(axis=1), o[c.acc_treated].mean(axis=1)
            pc, pt = pred[c.rna_control].mean(axis=1), pred[c.rna_treated].mean(axis=1)
            ev = (mc.notna() & mt.notna() & ((mc + mt) / 2 >= MIN_MEAN)).to_numpy()
            rows.append({"version": name, "dataset": c.dataset, "unit": c.unit, "condition": c.name, "n_enhancers": int(ev.sum()),
                         "nspe_before": nspe(mc.to_numpy()[ev], pc.to_numpy()[ev]), "nspe_after": nspe(mt.to_numpy()[ev], pt.to_numpy()[ev]),
                         "nspe_change": nspe((mt - mc).to_numpy()[ev], (pt - pc).to_numpy()[ev])})
            if name == "PERD-RF (RNA only)":
                frozen_change[c.name] = (pt - pc)
        print(f"  perturbations | {name}: median change NSPE {pd.DataFrame(rows).query('version == @name')['nspe_change'].median():.2f}", flush=True)
    return pd.DataFrame(rows), frozen_change


def pair_tables(d: dict, act: dict) -> pair_model.EnhancerTables:
    """Per-enhancer inputs of the pair model (motif z, ChIP binding, target genes, fixed properties), as scripts/20a."""
    enh = d["enhancers"].set_index("enhancer_id")
    ids, genes = enh.index, d["rna"].index
    mapping = pd.read_csv(TOY / "motif_archetype_tfs.csv")
    arch_tfs = {r.archetype: [t for t in str(r.tfs).split(";") if t in act["encode"].index] for r in mapping.itertuples()}
    arch_tfs = {a: v for a, v in arch_tfs.items() if v and a in d["motif_scale"].index}
    scale = d["motif_scale"].loc[sorted(arch_tfs)]
    arch_z = pd.DataFrame(np.tile((-scale["density_mean"] / scale["density_sd"]).to_numpy(), (len(ids), 1)), index=ids, columns=scale.index)
    m = d["motifs"][d["motifs"]["archetype"].isin(arch_z.columns)]
    arch_z_values = arch_z.to_numpy().copy()
    arch_z_values[arch_z.index.get_indexer(m["enhancer_id"]), arch_z.columns.get_indexer(m["archetype"])] = m["density_z"].to_numpy()
    arch_z = pd.DataFrame(arch_z_values, index=ids, columns=scale.index)
    scored = sorted({t for v in enh["tf_genes"] for t in v} & set(act["encode"].index))
    bound = pd.DataFrame([[float(t in set(v)) for t in scored] for v in enh["tf_genes"]], index=ids, columns=scored)
    gpos = {g: i for i, g in enumerate(genes)}
    pairs = [(i, gpos[g]) for i, tg in enumerate(enh["target_genes"]) for g in tg if g in gpos]
    targets = sparse.csr_matrix((np.ones(len(pairs), dtype=np.float32), ([p[0] for p in pairs], [p[1] for p in pairs])), shape=(len(ids), len(genes)))
    targets.sum_duplicates()
    targets.data[:] = 1.0
    keep = [c for c in d["openness"].columns if not c.startswith("SK-N-SH_treated")]
    o = d["openness"].loc[ids, keep]
    static = pd.DataFrame({"prior_mean": o.mean(axis=1), "prior_sd": o.std(axis=1), "length_kb": ((enh["end"] - enh["start"]) / 1000).clip(lower=MIN_KB),
                           "n_targets": np.diff(targets.indptr), "n_bound": bound.sum(axis=1)}, index=ids)
    return pair_model.EnhancerTables(ids, arch_z, bound, targets, genes, static, {a: arch_tfs[a] for a in arch_z.columns})


def q4_pair_model(d, act, frozen_change, n_jobs) -> pd.DataFrame:
    """Leave one perturbation (unit) out: train the shared forest on the others, predict the change of the held-out one."""
    t = pair_tables(d, act)
    names = t.feature_names()
    weights = pair_model.unit_weights(d["conditions"])
    per = {}
    for c in d["conditions"]:
        o = d["pert_open"].reindex(t.enhancers)
        control, treated = o[c.acc_control].mean(axis=1), o[c.acc_treated].mean(axis=1)
        d_rna = d["pert_rna"][c.rna_treated].mean(axis=1) - d["pert_rna"][c.rna_control].mean(axis=1)
        x = pair_model.condition_features(t, pair_model.delta_activity(act["pert"], c), d_rna, control)
        ev = (control.notna() & treated.notna() & ((control + treated) / 2 >= MIN_MEAN)).to_numpy()
        per[c.name] = (x, (treated - control).to_numpy(), ev)
    models = {"Pair model": np.arange(len(names)), "Starting openness only (control)": np.array([names.index(n) for n in ["baseline", *pair_model.STATIC]])}
    rows = []
    for unit in sorted({c.unit for c in d["conditions"]}):
        train = [c for c in d["conditions"] if c.unit != unit]
        x = np.vstack([per[c.name][0][per[c.name][2]] for c in train])
        y = np.concatenate([per[c.name][1][per[c.name][2]] for c in train])
        w = np.concatenate([np.full(per[c.name][2].sum(), weights[c.name] / per[c.name][2].sum()) for c in train])
        for label, cols in models.items():
            rf = RandomForestRegressor(n_estimators=500, max_features=max(1, len(cols) // 3), min_samples_leaf=PAIR_LEAF, random_state=SEED, n_jobs=n_jobs)
            rf.fit(x[:, cols], y, sample_weight=w)
            for c in (c for c in d["conditions"] if c.unit == unit):
                xc, yc, ev = per[c.name]
                rows.append({"model": label, "unit": unit, "condition": c.name, "nspe_change": nspe(yc[ev], rf.predict(xc[:, cols])[ev])})
        for c in (c for c in d["conditions"] if c.unit == unit):
            xc, yc, ev = per[c.name]
            rows.append({"model": "PERD-RF (RNA only)", "unit": unit, "condition": c.name, "nspe_change": nspe(yc[ev], frozen_change[c.name].reindex(t.enhancers).to_numpy()[ev])})
    return pd.DataFrame(rows)


def main() -> None:
    global TOY
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--out", type=Path, default=ROOT / "demo" / "output")
    parser.add_argument("--toy-dir", type=Path, default=TOY, help="folder with the toy data (default demo/toy_data)")
    args = parser.parse_args()
    TOY = args.toy_dir
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    d = load()
    print(f"toy data: {len(d['enhancers'])} enhancers, {d['rna'].shape[0]} genes x {d['rna'].shape[1]} ENCODE cells, "
          f"{len(d['conditions'])} perturbation conditions", flush=True)
    act = activities(d)
    print(f"TF activity: {act['encode'].shape[0]} TFs scored ({time.time() - start:.0f} s)", flush=True)
    versions = variants(d, act)

    print("Q1 + Q3a: unseen tissues", flush=True)
    q1 = q1_q3_unseen_tissues(d, versions, args.n_jobs)
    q1.to_csv(args.out / "q1_q3_unseen_tissues.csv", index=False)

    print("Q2 + Q3b: perturbations", flush=True)
    q2, frozen_change = q2_q3_perturbations(d, versions, args.n_jobs)
    q2.to_csv(args.out / "q2_q3_perturbation_nspe.csv", index=False)

    print("Q4: pair model, leave one perturbation out", flush=True)
    q4 = q4_pair_model(d, act, frozen_change, args.n_jobs)
    q4.to_csv(args.out / "q4_pair_model.csv", index=False)

    lines = ["Demo results (toy data)", "",
             "Q1/Q3a, unseen tissues (median per-cell NSPE; lower is better):", "",
             q1[["version", "scheme", "nspe_per_cell_median", "cross_cell_pearson_median"]].round(3).to_string(index=False), "",
             "Q2/Q3b, perturbations (median NSPE over conditions):", "",
             q2.groupby("version", sort=False)[["nspe_before", "nspe_after", "nspe_change"]].median().round(2).to_string(), "",
             "Q4, change NSPE per held-out perturbation (median over its conditions):", "",
             q4.groupby(["unit", "model"])["nspe_change"].median().unstack().round(2).to_string(), "",
             f"Total time: {(time.time() - start) / 60:.1f} min on {args.n_jobs} CPUs."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
