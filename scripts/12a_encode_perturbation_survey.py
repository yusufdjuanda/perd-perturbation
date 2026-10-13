"""Survey ENCODE for paired perturbation data (metadata only, no data download).

Finds released human DNase-seq / ATAC-seq and RNA-seq experiments whose biosamples carry a treatment, groups them
by (cell, treatment), and keeps combinations with both an accessibility and an RNA assay. For each: durations and
amounts per assay, number of experiments and biological replicates, whether untreated/vehicle experiments exist for
the same cell, and whether the cell appears to be among PERD's 167 ENCODE training samples.
Output: results/tables/encode_perturbation_survey.csv (+ the raw experiment list).
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PERD = ROOT.parent / "perd" / "data" / "interim"
TABLES = ROOT / "results" / "tables"
PORTAL = "https://www.encodeproject.org"
ASSAYS = {"DNase-seq": "accessibility", "ATAC-seq": "accessibility", "total RNA-seq": "rna", "polyA plus RNA-seq": "rna"}
FIELDS = [
    "accession", "assay_title", "status", "biosample_ontology.term_name", "biosample_ontology.classification",
    "replicates.biological_replicate_number", "replicates.library.biosample.treatments.treatment_term_name",
    "replicates.library.biosample.treatments.amount", "replicates.library.biosample.treatments.amount_units",
    "replicates.library.biosample.treatments.duration", "replicates.library.biosample.treatments.duration_units",
    "lab.title", "date_released", "assembly",
]


def search(assay: str) -> list[dict]:
    params = [("type", "Experiment"), ("assay_title", assay), ("status", "released"), ("replicates.library.biosample.donor.organism.scientific_name", "Homo sapiens"), ("format", "json"), ("limit", "all")]
    params += [("field", f) for f in FIELDS]
    url = PORTAL + "/search/?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "perd-perturbation/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)["@graph"]


def describe(experiment: dict) -> dict:
    treatments, reps = set(), set()
    for replicate in experiment.get("replicates", []):
        reps.add(replicate.get("biological_replicate_number"))
        for t in replicate.get("library", {}).get("biosample", {}).get("treatments", []):
            duration = f"{t.get('duration', '')}{(' ' + t.get('duration_units', '')) if t.get('duration') is not None else ''}".strip()
            amount = f"{t.get('amount', '')}{(' ' + t.get('amount_units', '')) if t.get('amount') is not None else ''}".strip()
            treatments.add((t.get("treatment_term_name", ""), amount, duration))
    return {
        "accession": experiment["accession"],
        "assay": experiment["assay_title"],
        "kind": ASSAYS[experiment["assay_title"]],
        "cell": experiment.get("biosample_ontology", {}).get("term_name", ""),
        "classification": experiment.get("biosample_ontology", {}).get("classification", ""),
        "treatments": sorted(treatments),
        "treatment_names": sorted({t[0] for t in treatments}),
        "n_bio_replicates": len([r for r in reps if r is not None]),
        "lab": experiment.get("lab", {}).get("title", ""),
        "assembly": ",".join(experiment.get("assembly", []) or []),
    }


def in_training(cell: str, training_names: list[str]) -> bool:
    key = re.sub(r"[^a-z0-9]", "", cell.lower())
    return any(key and key in re.sub(r"[^a-z0-9]", "", name.lower()) for name in training_names)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for assay in ASSAYS:
        hits = search(assay)
        print(f"{assay}: {len(hits)} released human experiments", flush=True)
        rows += [describe(e) for e in hits]
    experiments = pd.DataFrame(rows)
    experiments.assign(treatments=experiments["treatments"].map(json.dumps), treatment_names=experiments["treatment_names"].map(json.dumps)).to_csv(TABLES / "encode_perturbation_experiments.csv", index=False)

    training_names = list(pd.read_parquet(PERD / "rna_167_cells_symbol.parquet").columns)
    treated = experiments[experiments["treatment_names"].map(len) > 0]
    untreated_cells = {(c, k) for c, k in experiments.loc[experiments["treatment_names"].map(len) == 0, ["cell", "kind"]].itertuples(index=False)}
    survey = []
    for (cell, names), group in treated.groupby(["cell", treated["treatment_names"].map(tuple)]):
        kinds = set(group["kind"])
        if kinds != {"accessibility", "rna"}:
            continue
        info = {"cell": cell, "treatment": " + ".join(names), "classification": group["classification"].iloc[0]}
        for kind in ("accessibility", "rna"):
            sub = group[group["kind"] == kind]
            durations = sorted({t[2] for ts in sub["treatments"] for t in ts if t[2]})
            amounts = sorted({t[1] for ts in sub["treatments"] for t in ts if t[1]})
            info[f"{kind}_assays"] = ",".join(sorted(set(sub["assay"])))
            info[f"{kind}_experiments"] = len(sub)
            info[f"{kind}_durations"] = "; ".join(durations)
            info[f"{kind}_amounts"] = "; ".join(amounts)
            info[f"{kind}_median_bio_reps"] = float(sub["n_bio_replicates"].median())
        acc_d = set(info["accessibility_durations"].split("; ")) - {""}
        rna_d = set(info["rna_durations"].split("; ")) - {""}
        info["shared_durations"] = "; ".join(sorted(acc_d & rna_d))
        info["n_shared_durations"] = len(acc_d & rna_d)
        info["untreated_accessibility_exists"] = (cell, "accessibility") in untreated_cells
        info["untreated_rna_exists"] = (cell, "rna") in untreated_cells
        info["cell_in_perd_training_names"] = in_training(cell, training_names)
        info["labs"] = "; ".join(sorted(set(group["lab"])))
        survey.append(info)
    survey = pd.DataFrame(survey).sort_values(["n_shared_durations", "accessibility_experiments"], ascending=False)
    survey.to_csv(TABLES / "encode_perturbation_survey.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60):
        print(survey[["cell", "treatment", "accessibility_assays", "accessibility_experiments", "rna_experiments", "shared_durations", "accessibility_amounts", "accessibility_median_bio_reps", "rna_median_bio_reps", "untreated_accessibility_exists", "untreated_rna_exists", "cell_in_perd_training_names"]].to_string(index=False))
    print(f"\nPaired (cell, treatment) combinations: {len(survey)}")


if __name__ == "__main__":
    main()
