"""
analyze_kg_sources.py
=====================
Analyses and compares the quality and coverage of DrugBank and DrugCentral
knowledge graph components relative to the FAERS_TB vocabulary.

Reports:
  1. DrugBank KG (from knowledge_graph.pkl)
       - All edge types: node counts, edge counts, ID formats
       - Coverage of FAERS_TB drug vocab
       - Target, ADR, disease node breakdown

  2. DrugCentral (from drug.target.interaction.tsv)
       - Human-only drug-target pairs
       - Drug name → DrugBank ID mapping success rate
       - Coverage of FAERS_TB drug vocab
       - Action type breakdown
       - TB-specific drug coverage

  3. Overlap analysis
       - Drug node overlap (DrugBank IDs)
       - Protein target overlap (UniProt accessions)
       - Drug-target edge overlap

  4. FAERS_TB-filtered view
       - What each source contributes for the drugs actually in your model

Usage
-----
    cd /home/diyah/PhD_Research/ADR_EHR_Prediction_FAERS
    python scripts3/analyze_kg_sources.py
"""

import os
import sys
import pickle

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_PATH, PREPROCESSED_PATH, KG_EDGES, COL,
    DATASET_NAME,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_PATH           = os.path.join(PROJECT_ROOT,
                        "ADR_Prediction/data/processed_improved/knowledge_graph.pkl")
DC_PATH           = os.path.join(PROJECT_ROOT,
                        "ADR_Prediction/data/DrugCentral/drug.target.interaction.tsv")
MAPPING_PATH      = os.path.join(PROJECT_ROOT,
                        "ADR_Prediction/data/drug_id_mapping/drug-mappings.tsv")
FAERS_DRUGS_PATH  = os.path.join(DATA_PATH, "DRUGS_STANDARDIZED_DRUGBANK.csv")

# TB core drugs for targeted coverage check
TB_DRUGS = [
    "isoniazid", "rifampicin", "rifampin", "pyrazinamide", "ethambutol",
    "streptomycin", "rifabutin", "levofloxacin", "moxifloxacin", "linezolid",
    "bedaquiline", "delamanid", "ethionamide", "cycloserine", "capreomycin",
    "amikacin", "kanamycin",
]

SEP = "=" * 70


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n:,} / {total:,} ({100*n/total:.1f}%)"


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def subsection(title: str):
    print(f"\n  {'─'*60}")
    print(f"  {title}")
    print(f"  {'─'*60}")


# ── Load FAERS_TB drug vocabulary ─────────────────────────────────────────────

def load_faers_vocab() -> tuple[set, set, dict]:
    """
    Returns:
        faers_db_ids   : set of DrugBank IDs in FAERS_TB (e.g. 'DB00300')
        faers_names    : set of lowercase drug names in FAERS_TB
        name_to_db     : {drug_name_lower: DrugBank_ID}
    """
    df = pd.read_csv(FAERS_DRUGS_PATH, low_memory=False)
    df = df.dropna(subset=[COL["drugbank_id"]])
    faers_db_ids = set(df[COL["drugbank_id"]].unique())
    faers_names  = set(df[COL["drug_name"]].str.lower().str.strip().dropna())
    name_to_db   = (df.assign(_n=df[COL["drug_name"]].str.lower().str.strip())
                    .groupby("_n")[COL["drugbank_id"]].first().to_dict())
    return faers_db_ids, faers_names, name_to_db


# ── Section 1: DrugBank KG ────────────────────────────────────────────────────

def analyse_drugbank(faers_db_ids: set):
    section("1. DRUGBANK KNOWLEDGE GRAPH  (knowledge_graph.pkl)")

    with open(KG_PATH, "rb") as f:
        kg = pickle.load(f)

    edge_types = kg["edge_types"]
    stats      = kg.get("stats", {})

    # ── All edge types overview ───────────────────────────────────────────────
    subsection("1.1  All edge types")
    print(f"  {'Edge type':<25} {'Pairs':>10}  {'Unique src':>12}  {'Unique dst':>12}")
    print(f"  {'-'*65}")
    for etype, pairs in edge_types.items():
        pairs_list = list(pairs)
        srcs = {s for s, _ in pairs_list}
        dsts = {d for _, d in pairs_list}
        print(f"  {etype:<25} {len(pairs_list):>10,}  {len(srcs):>12,}  {len(dsts):>12,}")

    # ── Drug-target deep dive ─────────────────────────────────────────────────
    subsection("1.2  Drug-target edges (DrugBank)")
    dt_pairs  = list(edge_types.get("drug-target", []))
    db_drugs  = {s for s, _ in dt_pairs}
    db_targets= {d for _, d in dt_pairs}

    print(f"  Total drug-target pairs       : {len(dt_pairs):,}")
    print(f"  Unique drugs with targets      : {len(db_drugs):,}")
    print(f"  Unique protein targets         : {len(db_targets):,}")
    print(f"  Avg targets per drug           : {len(dt_pairs)/max(len(db_drugs),1):.1f}")

    # FAERS_TB coverage
    db_in_faers = db_drugs & faers_db_ids
    dt_faers_pairs = [(s, d) for s, d in dt_pairs if s in faers_db_ids]
    targets_faers  = {d for _, d in dt_faers_pairs}
    print(f"\n  ── FAERS_TB filtered ──")
    print(f"  FAERS_TB drugs total           : {len(faers_db_ids):,}")
    print(f"  FAERS_TB drugs with DB targets : {pct(len(db_in_faers), len(faers_db_ids))}")
    print(f"  Drug-target pairs (FAERS_TB)   : {len(dt_faers_pairs):,}")
    print(f"  Unique targets (FAERS_TB)      : {len(targets_faers):,}")
    print(f"  Avg targets per FAERS_TB drug  : {len(dt_faers_pairs)/max(len(db_in_faers),1):.1f}")

    # ── Drug-ADR (SIDER) ─────────────────────────────────────────────────────
    subsection("1.3  Drug-ADR edges (SIDER via DrugBank KG)")
    da_pairs  = list(edge_types.get("drug-adr", []))
    da_drugs  = {s for s, _ in da_pairs}
    da_adrs   = {d for _, d in da_pairs}
    da_faers  = [(s, d) for s, d in da_pairs if s in faers_db_ids]
    print(f"  Total drug-ADR pairs           : {len(da_pairs):,}")
    print(f"  Unique drugs                   : {len(da_drugs):,}")
    print(f"  Unique ADRs (UMLS CUIs)        : {len(da_adrs):,}")
    print(f"  Pairs covering FAERS_TB drugs  : {len(da_faers):,}")
    print(f"  FAERS_TB drugs with SIDER ADRs : {pct(len({s for s,_ in da_faers}), len(faers_db_ids))}")

    # ── Drug-drug (DDI) ───────────────────────────────────────────────────────
    subsection("1.4  Drug-drug interaction edges (DrugBank DDI)")
    dd_pairs  = list(edge_types.get("drug-drug", []))
    dd_drugs  = {s for s, _ in dd_pairs} | {d for _, d in dd_pairs}
    dd_faers  = [(s, d) for s, d in dd_pairs
                 if s in faers_db_ids and d in faers_db_ids]
    print(f"  Total DDI pairs                : {len(dd_pairs):,}")
    print(f"  Unique drugs in DDI            : {len(dd_drugs):,}")
    print(f"  DDI pairs within FAERS_TB      : {len(dd_faers):,}")

    # ── Disease edges ─────────────────────────────────────────────────────────
    subsection("1.5  Disease relation edges (KGIA-derived)")
    dc_pairs = list(edge_types.get("disease-comorbidity", []))
    dr_pairs = list(edge_types.get("disease-risk", []))
    print(f"  Disease-comorbidity pairs      : {len(dc_pairs):,}  "
          f"({len({s for s,_ in dc_pairs}|{d for _,d in dc_pairs}):,} unique diseases)")
    print(f"  Disease-risk pairs             : {len(dr_pairs):,}  "
          f"({len({s for s,_ in dr_pairs}|{d for _,d in dr_pairs}):,} unique nodes)")

    return db_drugs, db_targets, dt_faers_pairs


# ── Section 2: DrugCentral ────────────────────────────────────────────────────

def analyse_drugcentral(faers_db_ids: set, faers_names: set,
                         name_to_db: dict) -> tuple[set, set, list]:
    section("2. DRUGCENTRAL 2023  (drug.target.interaction.tsv)")

    dc = pd.read_csv(DC_PATH, sep="\t", dtype=str)
    dc.columns = dc.columns.str.strip('"')
    dc["DRUG_NAME"]   = dc["DRUG_NAME"].str.strip('"').str.lower().str.strip()
    dc["ACCESSION"]   = dc["ACCESSION"].str.strip('"').str.strip()
    dc["ACTION_TYPE"] = dc["ACTION_TYPE"].str.strip('"').str.strip()
    dc["ORGANISM"]    = dc["ORGANISM"].str.strip('"').str.lower()

    # ── Overview ──────────────────────────────────────────────────────────────
    subsection("2.1  Full dataset overview")
    print(f"  Total rows (all organisms)     : {len(dc):,}")
    dc_human = dc[dc["ORGANISM"].str.contains("homo sapiens", na=False)].copy()
    print(f"  Human-only rows                : {len(dc_human):,}")
    print(f"  Unique drugs (human)           : {dc_human['DRUG_NAME'].nunique():,}")
    print(f"  Unique targets / UniProt (human): {dc_human['ACCESSION'].nunique():,}")

    # Action type distribution
    subsection("2.2  Action type distribution (human)")
    action_counts = (dc_human["ACTION_TYPE"]
                     .fillna("(unspecified)")
                     .value_counts()
                     .head(12))
    print(f"  {'Action type':<35} {'Count':>8}  {'%':>6}")
    print(f"  {'-'*55}")
    total_human = len(dc_human)
    for action, count in action_counts.items():
        print(f"  {str(action):<35} {count:>8,}  {100*count/total_human:>5.1f}%")

    # ── Drug name → DrugBank ID mapping ──────────────────────────────────────
    subsection("2.3  Drug name → DrugBank ID mapping")

    # Build combined name→DB map
    all_name_to_db = dict(name_to_db)   # from FAERS
    if os.path.exists(MAPPING_PATH):
        mdf = pd.read_csv(MAPPING_PATH, sep="\t", dtype=str)
        mdf_valid = mdf.dropna(subset=["drugbankId", "name"])
        extra = (mdf_valid.assign(nl=mdf_valid["name"].str.lower().str.strip())
                 .set_index("nl")["drugbankId"].to_dict())
        # FAERS takes priority (already verified)
        extra.update(all_name_to_db)
        all_name_to_db = extra

    dc_human["DRUGBANK_ID"] = dc_human["DRUG_NAME"].map(all_name_to_db)
    matched   = dc_human.dropna(subset=["DRUGBANK_ID", "ACCESSION"]).copy()
    unmatched = dc_human[dc_human["DRUGBANK_ID"].isna()]["DRUG_NAME"].nunique()

    print(f"  DrugCentral unique drugs       : {dc_human['DRUG_NAME'].nunique():,}")
    print(f"  Mapped to a DrugBank ID        : "
          f"{pct(matched['DRUG_NAME'].nunique(), dc_human['DRUG_NAME'].nunique())}")
    print(f"  Unmapped drugs                 : {unmatched:,}")
    print(f"  Matched drug-target pairs      : {len(matched):,}")
    print(f"  (after dedup by DrugBank+UniProt)")
    matched_dedup = matched[["DRUGBANK_ID", "ACCESSION"]].drop_duplicates()
    print(f"  Unique (DrugBank, UniProt) pairs: {len(matched_dedup):,}")

    # ── FAERS_TB coverage ─────────────────────────────────────────────────────
    subsection("2.4  FAERS_TB coverage")
    matched_in_faers = matched_dedup[matched_dedup["DRUGBANK_ID"].isin(faers_db_ids)]
    dc_drugs_faers   = set(matched_in_faers["DRUGBANK_ID"])
    dc_targets_faers = set(matched_in_faers["ACCESSION"])

    print(f"  FAERS_TB drugs total           : {len(faers_db_ids):,}")
    print(f"  FAERS_TB drugs with DC targets : {pct(len(dc_drugs_faers), len(faers_db_ids))}")
    print(f"  Drug-target pairs (FAERS_TB)   : {len(matched_in_faers):,}")
    print(f"  Unique targets (FAERS_TB)      : {len(dc_targets_faers):,}")
    print(f"  Avg targets per FAERS_TB drug  : "
          f"{len(matched_in_faers)/max(len(dc_drugs_faers),1):.1f}")

    # ── TB-specific drug coverage ─────────────────────────────────────────────
    subsection("2.5  TB core drug coverage")
    print(f"  {'Drug':<20} {'In DrugCentral':>15}  {'#Targets':>10}  {'Action types'}")
    print(f"  {'-'*65}")
    for drug in sorted(TB_DRUGS):
        rows = matched_dedup[matched_dedup["DRUGBANK_ID"].isin(
            [all_name_to_db.get(drug, "")])]
        # Also check by drug name directly in dc_human
        dc_rows = matched[matched["DRUG_NAME"] == drug]
        found = "✓" if len(dc_rows) > 0 else "✗"
        n_tgt = dc_rows["ACCESSION"].nunique()
        actions = ", ".join(sorted(dc_rows["ACTION_TYPE"].dropna().unique())[:3])
        print(f"  {drug:<20} {found:>15}  {n_tgt:>10}  {actions}")

    return (set(matched_dedup["DRUGBANK_ID"]),
            set(matched_dedup["ACCESSION"]),
            list(zip(matched_in_faers["DRUGBANK_ID"], matched_in_faers["ACCESSION"])))


# ── Section 3: Overlap analysis ───────────────────────────────────────────────

def analyse_overlap(db_drugs: set, db_targets: set, db_faers_pairs: list,
                    dc_drugs: set, dc_targets: set, dc_faers_pairs: list,
                    faers_db_ids: set):
    section("3. OVERLAP ANALYSIS: DrugBank vs DrugCentral")

    # ── Drug node overlap ─────────────────────────────────────────────────────
    subsection("3.1  Drug node overlap (DrugBank IDs)")
    both_drugs = db_drugs & dc_drugs
    db_only    = db_drugs - dc_drugs
    dc_only    = dc_drugs - db_drugs
    print(f"  DrugBank drugs                 : {len(db_drugs):,}")
    print(f"  DrugCentral drugs (mapped)     : {len(dc_drugs):,}")
    print(f"  Shared drugs (both sources)    : {len(both_drugs):,}")
    print(f"  DrugBank-only                  : {len(db_only):,}")
    print(f"  DrugCentral-only               : {len(dc_only):,}")

    # ── Target node overlap ───────────────────────────────────────────────────
    subsection("3.2  Protein target overlap (UniProt accessions)")
    # Normalize DrugBank targets: strip 'PROTEIN_' prefix
    db_targets_norm = {t.replace("PROTEIN_", "") for t in db_targets}
    dc_targets_norm = set(dc_targets)   # already plain UniProt

    both_tgt = db_targets_norm & dc_targets_norm
    db_tgt_only = db_targets_norm - dc_targets_norm
    dc_tgt_only = dc_targets_norm - db_targets_norm
    print(f"  DrugBank unique targets        : {len(db_targets_norm):,}")
    print(f"  DrugCentral unique targets     : {len(dc_targets_norm):,}")
    print(f"  Shared protein targets         : {len(both_tgt):,}  "
          f"({100*len(both_tgt)/max(len(db_targets_norm|dc_targets_norm),1):.1f}% of union)")
    print(f"  DrugBank-only targets          : {len(db_tgt_only):,}")
    print(f"  DrugCentral-only targets       : {len(dc_tgt_only):,}")
    print(f"  Combined unique targets        : {len(db_targets_norm | dc_targets_norm):,}")

    # ── FAERS_TB filtered target overlap ─────────────────────────────────────
    subsection("3.3  FAERS_TB-filtered target overlap")
    db_tgt_f = {t.replace("PROTEIN_", "") for _, t in db_faers_pairs}
    dc_tgt_f = {t for _, t in dc_faers_pairs}
    both_f   = db_tgt_f & dc_tgt_f
    print(f"  DrugBank targets (FAERS_TB drugs)   : {len(db_tgt_f):,}")
    print(f"  DrugCentral targets (FAERS_TB drugs): {len(dc_tgt_f):,}")
    print(f"  Shared targets                      : {len(both_f):,}  "
          f"({100*len(both_f)/max(len(db_tgt_f),1):.1f}% of DrugBank targets)")
    print(f"  New targets from DrugCentral        : {len(dc_tgt_f - db_tgt_f):,}")
    print(f"  Combined unique targets (FAERS_TB)  : {len(db_tgt_f | dc_tgt_f):,}")

    # ── Edge overlap ──────────────────────────────────────────────────────────
    subsection("3.4  Drug-target edge overlap (FAERS_TB drugs)")
    # Normalize db pairs: strip PROTEIN_ prefix from target
    db_pairs_norm = {(d, t.replace("PROTEIN_", "")) for d, t in db_faers_pairs}
    dc_pairs_set  = set(dc_faers_pairs)

    both_edges = db_pairs_norm & dc_pairs_set
    db_only_e  = db_pairs_norm - dc_pairs_set
    dc_only_e  = dc_pairs_set  - db_pairs_norm
    print(f"  DrugBank drug-target pairs     : {len(db_pairs_norm):,}")
    print(f"  DrugCentral drug-target pairs  : {len(dc_pairs_set):,}")
    print(f"  Identical (drug, target) pairs : {len(both_edges):,}")
    print(f"  DrugBank-only edges            : {len(db_only_e):,}")
    print(f"  DrugCentral-only edges         : {len(dc_only_e):,}")
    print(f"  Combined unique edges          : {len(db_pairs_norm | dc_pairs_set):,}")


# ── Section 4: Summary table ──────────────────────────────────────────────────

def print_summary(db_drugs: set, db_targets: set, db_faers_pairs: list,
                  dc_drugs: set, dc_targets: set, dc_faers_pairs: list,
                  faers_db_ids: set):
    section("4. SUMMARY TABLE")
    db_tgt_f = {t.replace("PROTEIN_", "") for _, t in db_faers_pairs}
    dc_tgt_f = {t for _, t in dc_faers_pairs}
    db_tgt_norm = {t.replace("PROTEIN_", "") for t in db_targets}

    rows = [
        ("Metric",                        "DrugBank",             "DrugCentral"),
        ("─"*30,                          "─"*20,                 "─"*20),
        ("Total drug nodes",              f"{len(db_drugs):,}",   f"{len(dc_drugs):,}"),
        ("Total target nodes",            f"{len(db_tgt_norm):,}",f"{len(dc_targets):,}"),
        ("Total drug-target edges",       f"{sum(1 for _ in db_faers_pairs)*0+len(list(db_faers_pairs_raw)):,}",
                                                                    f"{len(dc_faers_pairs):,}"),
        ("FAERS_TB drugs covered",        pct(len({d for d,_ in db_faers_pairs}), len(faers_db_ids)),
                                          pct(len({d for d,_ in dc_faers_pairs}), len(faers_db_ids))),
        ("Targets for FAERS_TB drugs",    f"{len(db_tgt_f):,}",   f"{len(dc_tgt_f):,}"),
        ("Edges for FAERS_TB drugs",      f"{len(db_faers_pairs):,}", f"{len(dc_faers_pairs):,}"),
        ("Combined unique targets",       "─",                    f"{len(db_tgt_f|dc_tgt_f):,}"),
        ("New targets vs DrugBank",       "─",                    f"{len(dc_tgt_f - db_tgt_f):,}"),
        ("Last updated",                  "DrugBank 5.1 (2021)",  "DrugCentral 2023"),
        ("ID system (drugs)",             "DrugBank ID",          "Name → DrugBank ID"),
        ("ID system (targets)",           "PROTEIN_UniProt",      "UniProt (normalised)"),
    ]

    print(f"  {'Metric':<35} {'DrugBank':>22} {'DrugCentral':>22}")
    print(f"  {'─'*80}")
    for r in rows[2:]:
        print(f"  {r[0]:<35} {r[1]:>22} {r[2]:>22}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{SEP}")
    print(f"  KG SOURCE QUALITY ANALYSIS")
    print(f"  Dataset: {DATASET_NAME}")
    print(SEP)

    if not os.path.exists(FAERS_DRUGS_PATH):
        print(f"ERROR: FAERS drugs file not found at {FAERS_DRUGS_PATH}")
        sys.exit(1)
    if not os.path.exists(KG_PATH):
        print(f"ERROR: knowledge_graph.pkl not found at {KG_PATH}")
        sys.exit(1)
    if not os.path.exists(DC_PATH):
        print(f"ERROR: DrugCentral TSV not found at {DC_PATH}")
        sys.exit(1)

    faers_db_ids, faers_names, name_to_db = load_faers_vocab()
    print(f"\n  FAERS_TB vocabulary loaded: {len(faers_db_ids):,} drugs")

    db_drugs, db_targets, db_faers_pairs = analyse_drugbank(faers_db_ids)
    dc_drugs, dc_targets, dc_faers_pairs = analyse_drugcentral(
        faers_db_ids, faers_names, name_to_db)
    analyse_overlap(db_drugs, db_targets, db_faers_pairs,
                    dc_drugs, dc_targets, dc_faers_pairs, faers_db_ids)

    # Summary — inline counts to avoid variable scoping issue
    section("4. SUMMARY TABLE")
    db_tgt_f   = {t.replace("PROTEIN_", "") for _, t in db_faers_pairs}
    dc_tgt_f   = {t for _, t in dc_faers_pairs}
    db_tgt_norm= {t.replace("PROTEIN_", "") for t in db_targets}

    rows = [
        ("Total drug nodes (full KG)",    f"{len(db_drugs):,}",    f"{len(dc_drugs):,}"),
        ("Total target nodes (full KG)",  f"{len(db_tgt_norm):,}", f"{len(dc_targets):,}"),
        ("FAERS_TB drugs covered",
            pct(len({d for d,_ in db_faers_pairs}), len(faers_db_ids)),
            pct(len({d for d,_ in dc_faers_pairs}), len(faers_db_ids))),
        ("Drug-target edges (FAERS_TB)",  f"{len(db_faers_pairs):,}", f"{len(dc_faers_pairs):,}"),
        ("Unique targets (FAERS_TB)",     f"{len(db_tgt_f):,}",    f"{len(dc_tgt_f):,}"),
        ("Combined unique targets",       "─",                     f"{len(db_tgt_f|dc_tgt_f):,}"),
        ("New targets vs DrugBank",       "─",                     f"{len(dc_tgt_f - db_tgt_f):,}"),
        ("Shared (drug, target) edges",   "─",
            f"{len({(d,t.replace('PROTEIN_','')) for d,t in db_faers_pairs} & set(dc_faers_pairs)):,}"),
        ("Last updated",                  "DrugBank 5.1 (2021)",   "DrugCentral 2023"),
        ("Drug ID system",                "DrugBank ID",           "Name → DrugBank ID"),
        ("Target ID system",              "PROTEIN_<UniProt>",     "Plain UniProt"),
    ]

    print(f"\n  {'Metric':<38} {'DrugBank':>22} {'DrugCentral':>22}")
    print(f"  {'─'*85}")
    for label, db_val, dc_val in rows:
        print(f"  {label:<38} {db_val:>22} {dc_val:>22}")

    print(f"\n{SEP}")
    print(f"  Analysis complete.")
    print(SEP)


if __name__ == "__main__":
    main()
