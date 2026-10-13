"""Post hoc (user-approved, 2026-10-08): PERD-RF input variants predicting every perturbation dataset.

tfadd (as 14b), tfmotif and tfboth (feature lists as 19a), fit on the 167 ENCODE cells; prediction passes as frozen PERD:
A549 alone; K562 + T cells (union of genes, missing = 0); HCT116 + SK-N-SH (likewise). RF only. No openness of a test
sample is read. Usage: python scripts/21a_variant_perturbation_predictions.py tfadd|tfmotif|tfboth [pass,...]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from perd_perturbation import tf_activity
from perd_perturbation.perd_rf import run_perd_model

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
ACT = PROCESSED / "tf_activity"
TABLES = ROOT / "results" / "tables"
PASSES = {"a549": {"dex": "a549_dex"}, "k562_tcell": {"k562": "k562", "tcell": "tcell"}, "hct116_sknsh": {"hct116": "hct116", "sknsh": "sknsh"}}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


motif = load_module("motif", ROOT / "scripts" / "19a_motif_activity_heldout.py")


def main() -> None:
    variant = sys.argv[1]
    if variant not in ("tfadd", "tfmotif", "tfboth"):
        raise SystemExit("usage: 21a_variant_perturbation_predictions.py tfadd|tfmotif|tfboth [pass,...]")
    passes = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PASSES)
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    start = time.time()
    encode_rna = pd.read_parquet(PERD / "rna_167_cells_symbol.parquet")
    openness = pd.read_parquet(PERD / "enhancer_openness.parquet")
    frozen_training = pd.read_parquet(PERD / "netopen_training_enhancers.parquet")
    enhancers = pd.Index(frozen_training["enhancer_id"])
    act_encode = pd.read_parquet(ACT / "activity_encode167.parquet")
    binding = pd.read_parquet(PERD / "enhancer_tf_binding.parquet").set_index("enhancer_id")["tf_genes"]
    targets = pd.read_parquet(PERD / "genehancer_enhancers.parquet").drop_duplicates("enhancer_id").set_index("enhancer_id")["target_genes"]

    use_act = variant in ("tfadd", "tfboth")
    use_arch = variant in ("tfmotif", "tfboth")
    if use_act:
        training = tf_activity.feature_lists(frozen_training, binding, targets, set(act_encode.index), "tfadd")[["enhancer_id", "usable_genes"]]
        train_features = tf_activity.with_activity(encode_rna, act_encode)
    else:
        training = frozen_training[["enhancer_id", "usable_genes"]].copy()
        train_features = encode_rna
    if use_arch:
        arch_encode = motif.archetype_activity(act_encode)
        sets = motif.motif_sets(enhancers, set(arch_encode.index))
        training["usable_genes"] = [list(g) + [motif.ARCH_PREFIX + a for a in sets.get(e, [])] for e, g in zip(training["enhancer_id"], training["usable_genes"])]
        train_features = pd.concat([train_features, arch_encode.rename(index=lambda a: motif.ARCH_PREFIX + a)[encode_rna.columns]], axis=0)
    print(f"{variant}: median inputs per enhancer {training['usable_genes'].map(len).median():.0f}; {time.time() - start:.0f}s", flush=True)

    summary = {"variant": variant}
    for name in passes:
        t0 = time.time()
        datasets = PASSES[name]
        rna = {d: pd.read_parquet(PROCESSED / f"encode_{d}" / "rna_log2fpkm_replicates.parquet") for d in datasets}
        if len(rna) == 1:
            profiles = next(iter(rna.values()))
        else:
            genes = next(iter(rna.values())).index
            for t in list(rna.values())[1:]:
                genes = genes.union(t.index)
            profiles = pd.concat([t.reindex(genes).fillna(0.0) for t in rna.values()], axis=1)
        act = pd.concat([pd.read_parquet(ACT / f"activity_{a}.parquet") for a in datasets.values()], axis=1)[profiles.columns]
        feats = tf_activity.with_activity(profiles, act) if use_act else profiles
        if use_arch:
            feats = pd.concat([feats, motif.archetype_activity(act).rename(index=lambda a: motif.ARCH_PREFIX + a)[profiles.columns]], axis=0)
        table = run_perd_model(training, train_features, openness, feats, model="rf", n_jobs=n_jobs)
        meta = ["n_genes_train", "n_genes_used", "skip_reason"]
        for d, t in rna.items():
            table[list(t.columns) + meta].to_parquet(PROCESSED / f"encode_{d}" / f"perd_predictions_{variant}_rf.parquet")
        summary[name] = {"profiles": int(profiles.shape[1]), "fitted": int((table["skip_reason"] == "").sum()), "minutes": (time.time() - t0) / 60}
        print(name, summary[name], flush=True)
    (TABLES / f"a11_variant_predictions_{variant}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"total {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
