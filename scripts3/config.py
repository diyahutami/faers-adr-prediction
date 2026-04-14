"""
config.py
=========
Shared configuration for the PreciseADR reproduction pipeline - scripts3 version.
Supports multiple dataset variants: FAERS_ALL, FAERS_TB, FAERS_TB_DRUGS

Update DATASET_NAME to select which dataset to use.
"""

import os

# ── Dataset Selection ──────────────────────────────────────────────────────
# Choose which dataset to use: "FAERS_ALL", "FAERS_TB", or "FAERS_TB_DRUGS"
DATASET_NAME = os.environ.get("DATASET_NAME", "FAERS_ALL")  # Can be set via environment variable

# ── Paths ──────────────────────────────────────────────────────────────────
# Get the project root directory (parent of scripts3/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset-specific data paths
DATA_PATH = os.path.join(_PROJECT_ROOT, f"data/{DATASET_NAME}")
PREPROCESSED_PATH = os.path.join(_PROJECT_ROOT, f"data/preprocessed_{DATASET_NAME.lower()}")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, f"output_{DATASET_NAME.lower()}")

# Reference data paths (shared across datasets)
DRUGBANK_MAPPING_FILE_PATH = os.path.join(_PROJECT_ROOT, "data/drugname_drugbank_mapping.tsv")
MEDDRA_MAPPING_FILE_PATH = os.path.join(_PROJECT_ROOT, "data/meddra.tsv")

# Sub-directories (created automatically by each step)
GRAPH_PATH = os.path.join(OUTPUT_PATH, f"graphs_{DATASET_NAME}")
MODEL_PATH = os.path.join(OUTPUT_PATH, f"models_{DATASET_NAME}")
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}")

# ── Separator used in FAERS text files ────────────────────────────────────
FAERS_SEP = ","

# ── Table filenames (relative to DATA_PATH) ──────────────────────────────
#   Use the enriched versions if DrugBank / MedDRA mapping has been run.
TABLES = {
    "demographics"           : "DEMOGRAPHICS.csv",
    "drugs_standardized"     : "DRUGS_STANDARDIZED_DRUGBANK.csv",  # fallback: DRUGS_STANDARDIZED.csv
    "adverse_reactions"      : "ADVERSE_REACTIONS.csv",
    "drug_indications"       : "DRUG_INDICATIONS.csv",
    "therapy_dates"          : "THERAPY_DATES.csv",
    "case_outcomes"          : "CASE_OUTCOMES.csv",
    "report_sources"         : "REPORT_SOURCES.csv",
    "proportionate_analysis" : "PROPORTIONATE_ANALYSIS.csv",
    "contingency_table"      : "CONTINGENCY_TABLE.csv",
}

# ── Column names ─────────────────────────────────────────────────────────
COL = {
    # DEMOGRAPHICS
    "primaryid"    : "PRIMARYID",
    "age"          : "AGE",
    "gender"       : "GENDER",
    "country_code" : "COUNTRY_CODE",
    "period"       : "PERIOD",
    # DRUGS_STANDARDIZED
    "rxcui"        : "RXCUI",
    "drug_name"    : "DRUG",
    "drugbank_id"  : "DRUGBANK_ID",
    # ADVERSE_REACTIONS
    "adverse_event": "ADVERSE_EVENT",
    "meddra_code"  : "MEDDRA_CODE",
    # DRUG_INDICATIONS
    "indication"   : "DRUG_INDICATION",
    # CASE_OUTCOMES
    "outc_cod"     : "OUTC_COD",
    # REPORT_SOURCES
    "rpsr_cod"     : "RPSR_COD",
    # PROPORTIONATE_ANALYSIS
    "ror"          : "ROR",
    "ror_lb"       : "ROR_LB",
    "ic"       : "IC",
    "ic025"    : "IC025",
    "ic975"    : "IC975",
    "prr"      : "PRR",
    "pt_code"  : "PT_CODE",
    "obs_count": "A", # observed count (cell A for contingency table)
}

# ── Pharmacovigilance signal edges (drug → ADR direct edges) ─────────────────
# Adds (drug, signals_adr, adr) edges for statistically significant drug-ADR pairs.
# Edge features: [log_ror, ic, ror_lb_flag, log_a]  (4-dim per edge)
#   log_ror     = log(clip(ROR, 0.01, ror_clip) + 1)   signal strength
#   ic          = clip(IC, -ic_clip, ic_clip)           Bayesian IC (BCPNN)
#   ror_lb_flag = float(ROR_LB > 1)                     ROR lower bound significant
#   log_a       = log(A + 1)                            observed count (reliability)
# Filtering: pairs with A >= min_obs_count AND IC025 > ic025_threshold only.
SIGNAL_EDGES = {
    "enabled"         : False,
    "min_obs_count"   : 3,      # minimum observed cases per pair (removes very noisy pairs)
    "ic025_threshold" : 0.0,    # IC025 > 0  ↔  Bayesian lower credibility bound significant
    "ror_clip"        : 50.0,
    "ic_clip"         : 5.0,
    "edge_dim"        : 4,      # [log_ror, ic, ror_lb_flag, log_a]
}

# ── External Knowledge Graph integration ──────────────────────────────────────
# Integrates biomedical knowledge from DrugBank, SIDER, and KGIA into the FAERS
# heterogeneous graph.  All three steps use the pre-processed knowledge_graph.pkl
# produced by ADR_Prediction/scripts_improved/ (already on disk).
#
# Step 1  use_drug_target  — (drug, has_target, target) edges from DrugBank.
#           Adds a NEW "target" node type (UniProt proteins).  Drugs sharing
#           molecular targets tend to share ADR profiles → mechanistic pathway.
#           Drug IDs: DrugBank ID (100 % overlap with FAERS_TB vocab).
#
# Step 2  use_drugbank_ddi — (drug, db_interacts, drug) edges from DrugBank.
#           Pharmacological DDI (real interactions), stored as a SEPARATE edge
#           type from the FAERS co-occurrence "interacts" edges.
#           Toggle GRAPH["include_ddi"] independently to ablate each source.
#
# Step 3  use_sider_adr   — (drug, kg_has_adr, adr) edges from SIDER.
#           Population-level drug-ADR prior knowledge, separate from the
#           patient-level FAERS "experiences" edges.
#           ADR mapping: SIDER UMLS CUI → MedDRA code via meddra.tsv (100 %).
#           Coverage: 58.6 % of FAERS_TB MedDRA codes are present in SIDER.
KG_EDGES = {
    "enabled"          : True,
    "kg_data_path"     : os.path.join(_PROJECT_ROOT,
                             "ADR_Prediction/data/processed_improved/knowledge_graph.pkl"),
    "meddra_path"      : MEDDRA_MAPPING_FILE_PATH,   # shared meddra.tsv (UMLS→MedDRA)
    "use_drug_target"  : False,   # Step 1: new target node type + drug→target edges
    "use_drugbank_ddi" : False,  # Step 2: DrugBank DDI (db_interacts edge type)
    "use_sider_adr"    : False,  # Step 3: SIDER drug-ADR prior (kg_has_adr edge type)
    # ── KGIA disease relations ──────────────────────────────────────────────
    # Adds disease-disease edges from the KGIA biomedical knowledge graph.
    # All KGIA disease IDs are UMLS CUIs; mapped to MedDRA via meddra.tsv.
    # Only pairs where BOTH disease nodes exist in disease_vocab are kept.
    #
    # Step 4  use_kgia_comorbidity — (disease, kg_comorbid, disease)
    #           Known comorbidity pairs from KGIA (101,071 bidirectional pairs).
    #           Clinically relevant for TB: HIV, diabetes, hepatitis comorbidities
    #           are already present and affect ADR profiles.
    #
    # Step 5  use_kgia_risk       — (disease, kg_has_risk, disease)
    #           Disease-risk factor edges from KGIA (34,423 directed pairs).
    #           Encodes "disease X predisposes to condition Y" (e.g., TB→hepatitis).
    "kgia_data_path"      : os.path.join(_PROJECT_ROOT,
                                "ADR_Prediction/data/KGIA/biomedical_KG.txt"),
    "use_kgia_comorbidity": False,  # Step 4: KGIA disease comorbidity edges
    "use_kgia_risk"       : False,  # Step 5: KGIA disease-risk factor edges
    # ── DrugCentral drug-target integration ────────────────────────────────
    # Adds (drug, dc_has_target, target) edges from DrugCentral 2023.
    # DrugCentral uses UniProt accessions (same protein space as DrugBank).
    # Targets are merged into the shared target_vocab (PROTEIN_<UniProt> format).
    # Drug mapping: DrugCentral drug name → DrugBank ID via drug_id_mapping.tsv
    #   + direct FAERS drug name lookup. Coverage: 838/843 FAERS_TB drugs.
    # Adds 6,922 drug-target pairs; ~439 targets new beyond DrugBank.
    #
    # Step 6  use_drugcentral_target — (drug, dc_has_target, target)
    #           Separate edge type from DrugBank (drug, has_target, target) so
    #           each source can be ablated independently.
    "use_drugcentral_target"   : False,  # Step 6: DrugCentral drug-target edges
    "drugcentral_data_path"    : os.path.join(_PROJECT_ROOT,
                                     "ADR_Prediction/data/DrugCentral/drug.target.interaction.tsv"),
    "drugcentral_mapping_path" : os.path.join(_PROJECT_ROOT,
                                     "ADR_Prediction/data/drug_id_mapping/drug-mappings.tsv"),
    # ── OnSIDES drug-label ADE prior ───────────────────────────────────────────
    # Adds (drug, onsides_has_adr, adr) edges from the OnSIDES v3.1 database.
    # OnSIDES extracts ADEs from FDA Structured Product Labels using PubMedBERT,
    # providing a high-quality label-derived drug-ADE knowledge base complementary
    # to FAERS (spontaneous reports) and SIDER (older label-derived database).
    #
    # ID alignment (direct, no bridging table needed):
    #   Drug : OnSIDES ingredient_id = RxNorm RXCUI (ingredient-level)
    #          → matches FAERS RXCUI directly when drug_vocab is RXCUI-keyed,
    #            or via RXCUI→DrugBank_ID reverse map when DrugBank-keyed.
    #   ADR  : OnSIDES effect_meddra_id = MedDRA PT code (numeric)
    #          → matches FAERS MEDDRA_CODE / adr_vocab directly.
    #
    # Coverage (high_confidence.csv, 663 pairs, 71 ingredient RXCUIs, 180 MedDRA PTs):
    #   vs FAERS_TB: 60/71 OnSIDES drugs match (85 %), 166/180 ADRs match (92 %)
    #
    # Step 7  use_onsides_adr — (drug, onsides_has_adr, adr)
    #           Separate edge type from SIDER (kg_has_adr) so each source can be
    #           ablated independently.  Toggle in run_kg_ablation.py (ablation K).
    "use_onsides_adr"          : False,  # Step 7: OnSIDES drug-label ADE prior
    "onsides_data_path"        : os.path.join(_PROJECT_ROOT,
                                     "ADR_Prediction/data/ONSIDES/dataset-onsides-v3.1.0/csv"),
}

# ── Quality-control settings ──────────────────────────────────────────────
# Note: For TB datasets, data is already filtered by TB criteria
QC = {
    "country_filter"  : "US",
    "reporter_filter" : "HP",
    # "min_frequency"   : 100,      # for FAERS_ALL
    "min_frequency" : 200             # for FAERS_TB and FAERS_TB_DRUGS (already filtered, so can use lower threshold)
}

# ── Age groups ────────────────────────────────────────────────────────────
AGE_BINS   = [0,   18,  65,  121]          # right-exclusive upper bounds
AGE_LABELS = ["Youth", "Adult", "Elderly"]

# ── Association mining ────────────────────────────────────────────────────
ASSOC = {
    "alpha"           : 0.05,
    "min_cooccurrence": 10,
    "correction"      : "fdr_bh",
}

# ── Train / val / test split ──────────────────────────────────────────────
SPLIT = {
    "train": 0.75,
    "val"  : 0.125,
    "test" : 0.125,
    "seed" : 42,
}

# ── ADR label filtering ───────────────────────────────────────────────────
# Controls which ADRs are included in the label space (vocabulary).
# Set top_n_adrs to an integer to keep only the N most-frequent ADRs.
# Set to None to keep all ADRs that pass QC-3.
# adr_list_file: optional path to a .txt file (one ADR name per line) produced
#   by analyze_adr_frequency.py --save-list. If set, it overrides top_n_adrs.
ADR_FILTER = {
    "top_n_adrs"    : None,        # keep top-50 most frequent ADRs (set None to keep all)
    # Rationale: full vocab has 1,646 ADRs at 0.2% label density (median 3 patients/class),
    # which is unlearnable. Top-50 ADRs have 200–1,136 patients each (1.5–8.3% prevalence),
    # matching the paper's ~100 samples/class regime where AUC reaches 0.65–0.75 (Fig. 5a).
    "adr_list_file" : None    # e.g. "data/preprocessed_faers_tb/top_50_adrs.txt"
}

# ── DDI / III edge configuration ──────────────────────────────────────────
# include_ddi : True | False   — whether to add drug-drug co-occurrence edges
# include_iii : True | False   — whether to add disease-disease co-occurrence edges
# edge_weight_mode:
#   "log"   → weight = log(count + 1)   [original behaviour]
#   "raw"   → weight = count            [raw co-occurrence count]
#   "none"  → no edge_attr stored       [topology only; same as what HGTConv sees]
# threshold : minimum co-occurrence count to create an edge
GRAPH = {
    "include_ddi"       : True,     # ablation showed G9 (both) is catastrophic; default off
    "include_iii"       : True,
    "edge_weight_mode"  : "log",     # "log" | "raw" | "none"
    "ddi_threshold"     : 3,
    "iii_threshold"     : 3,
}

# ── Model hyperparameters ─────────────────────────────────────────────────
MODEL = {
    "embedding_dim"  : 128,          # default: 256  # best: 128  # 256-dim tested: no gain (+0.0005 AUC, more overfitting)
    "num_hgt_layers" : 2,           # default: 3    # best: 4
    "num_heads"      : 2,           # default: 4    # best: 2   # 4 heads tested with 256-dim
    "dropout"        : 0.1,         # default: 0.3  # best: 0.1         
    "focal_gamma"    : 2.0,         # default: 2.0  # confirmed best (γ=2 Hit@5 +14% vs γ=1)
    "batch_size"     : 512,         # default: 512
    "lr"             : 1e-3,        # default: 5e-3 # best: 1e-3
    "weight_decay"   : 0.0,         # NOTE: 1e-4 tested but kills learning (focal gradients too small vs L2 penalty);
                                    #   architecture already regularised via HGT dropout + augmentation dropout
    "max_epochs"     : 300,         # was 500; model never improves past epoch ~50 so 300 is sufficient
    "patience"       : 100,          # was 300 (wasteful); 75 gives headroom past epoch-50 peak without burning GPU
    "patient_chunk_size": 4096,     # patients per chunk in loss/eval; reduce if OOM (e.g. 1024 for 8GB GPU + 582 ADRs)
    "use_gradient_checkpointing": True,  # gradient checkpoint HGT layers; required for FAERS_ALL (200k nodes)
                                         # trades ~2× compute for ~70% less GPU memory (no per-edge activations stored)
    # NEW PARAMETER FOR MODEL_V2
    "edge_drop_prob" : 0.1,
    "projection_dim" : 128,
    "projection_dropout" : 0.1,
    "similarity_threshold": 0.3,  # Jaccard similarity threshold for SupCon positive pairs (30% overlap)
    "alpha_range"    : [0.0, 0.01, 0.05, 0.1, 0.3],   # finer search near 0 where val evidence points
    "tau_range"      : [0.01, 0.05, 0.1],              # 15 combinations total (was 30)
    # ── Per-ADR positive weighting (Option C / REC-12) ───────────────────────
    "use_pos_weight" : True,    # weight rare ADRs more heavily in Focal Loss
    "pos_weight_cap" : 50.0,    # clamp max weight (avoids extreme values for ADRs
                                #   with only 1–2 positives in training)
    "alpha"          : 0.0,   # default; overridden by grid search
    # alpha=0.0 outperforms alpha=0.1 on this dataset: with 10k patients and 50 ADRs,
    # InfoNCE at alpha=0.1 pulls NCE loss to near-zero (0.23), causing contrastive
    # overfitting (val→test AUC gap 0.024 vs 0.010, Hit@10 0.4606 vs 0.5472).
    # Use grid search to find optimal alpha across [0.0, 0.1, 0.3, 0.5, 0.7, 0.9].
    "tau"            : 0.05,
    # InfoNCE configuration (per paper Equation 10)
    "use_sampled_negatives": False,   # False = exact paper formula (may OOM), True = memory-efficient
    "max_negatives"        : None,   # Maximum negative samples when use_sampled_negatives=True
    # Memory optimization settings for T4 GPU (15GB)
    "use_amp"              : True,    # Use automatic mixed precision (FP16) training
    "accumulation_steps"   : 4       # Gradient accumulation steps (increase if still OOM)
}

# ── Evaluation ────────────────────────────────────────────────────────────
EVAL = {
    "hit_k"  : [1, 2, 5, 10, 20],
    "ndcg_k" : [5, 10, 20],
}

# ── Ablation ─────────────────────────────────────────────────────────────
ABLATION = {
    "samples_per_class": [10, 20, 30, 50, 70, 100],
    "n_seeds"          : 3,
    "n_gender_adrs"    : 20,
    "n_age_adrs"       : 20,
}

# ── Outcome codes ─────────────────────────────────────────────────────────
OUTCOME_CODES = ["DE", "LT", "HO", "DS", "CA", "RI", "OT"]

# ── Dataset information ───────────────────────────────────────────────────
DATASET_INFO = {
    "FAERS_ALL": {
        "description": "Full FAERS dataset (2015-2025)",
        "source": "data/standardized_faers_2015_2025",
    },
    "FAERS_TB": {
        "description": "Tuberculosis-related cases",
        "source": "data/standardized_faers_2015_2025 (filtered by FAERS_TB_PRIMARYID_SET)",
    },
    "FAERS_TB_DRUGS": {
        "description": "TB cases with TB-specific drugs",
        "source": "data/standardized_faers_2015_2025 (filtered by FAERS_TB_DRUGS_PRIMARYID_SET)",
    },
}

def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Dataset:       {DATASET_NAME}")
    print(f"Description:   {DATASET_INFO[DATASET_NAME]['description']}")
    print(f"Data Path:     {DATA_PATH}")
    print(f"Preprocessed:  {PREPROCESSED_PATH}")
    print(f"Output:        {OUTPUT_PATH}")
    print(f"Graphs:        {GRAPH_PATH}")
    print(f"Models:        {MODEL_PATH}")
    print(f"Results:       {RESULTS_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
