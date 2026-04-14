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
DATASET_NAME = os.environ.get("DATASET_NAME", "FAERS_TB")  # Can be set via environment variable

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
KG_EDGES = {
    "enabled"           : True,
    "kg_data_path"      : os.path.join(_PROJECT_ROOT, "ADR_Prediction/data/processed_improved/knowledge_graph.pkl"),
    "meddra_path"       : MEDDRA_MAPPING_FILE_PATH,   # shared meddra.tsv (UMLS→MedDRA)
    "use_drug_target"   : False,  # Step 1: new target node type + drug→target edges — (drug, has_target, target) from DrugBank
    "use_drugbank_ddi"  : False,  # Step 2: DrugBank DDI — (drug, db_interacts, drug) edges from DrugBank.
    "use_sider_adr"     : False,  # Step 3: SIDER drug-ADR prior — (drug, kg_has_adr, adr) edges from SIDER.
    "kgia_data_path"    : os.path.join(_PROJECT_ROOT, "ADR_Prediction/data/KGIA/biomedical_KG.txt"),
    "use_kgia_comorbidity"  : False,  # Step 4: KGIA disease comorbidity edges (disease, kg_comorbid, disease)
    "use_kgia_risk"         : False,  # Step 5: KGIA disease-risk factor edges (disease, kg_has_risk, disease)
    "drugcentral_data_path" : os.path.join(_PROJECT_ROOT, "ADR_Prediction/data/DrugCentral/drug.target.interaction.tsv"),
    "drugcentral_mapping_path"  : os.path.join(_PROJECT_ROOT, "ADR_Prediction/data/drug_id_mapping/drug-mappings.tsv"),
    "use_drugcentral_target"    : False,  # Step 6: DrugCentral drug-target edges (drug, dc_has_target, target)
    "onsides_data_path" : os.path.join(_PROJECT_ROOT, "ADR_Prediction/data/ONSIDES/dataset-onsides-v3.1.0/csv"),
    "use_onsides_adr"   : False,  # Step 7: OnSIDES drug-label ADE prior (drug, onsides_has_adr, adr)
}

# ── Quality-control settings ──────────────────────────────────────────────
# Note: For TB datasets, data is already filtered by TB criteria
QC = {
    "country_filter"  : "US",
    "reporter_filter" : "HP",
    # "min_frequency"   : 100,      # for FAERS_ALL
    "min_frequency" : 2             # for FAERS_TB and FAERS_TB_DRUGS (already filtered, so can use lower threshold)
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
    "top_n_adrs"    : 50,        # keep top-50 most frequent ADRs (set None to keep all)
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
    "embedding_dim"  : 128,         # paper value: 256; FAERS_TB best was 128 but FAERS_ALL (200k patients) needs more capacity
    "num_hgt_layers" : 3,           # paper value: 3
    "num_heads"      : 4,           # paper value: 4; head_dim = 256/4 = 64
    "dropout"        : 0.1,         # default: 0.3  # best: 0.1
    "focal_gamma"    : 2.0,         # confirmed best (γ=2 Hit@5 +14% vs γ=1)
    "batch_size"     : 512,         # default: 512
    "lr"             : 1e-3,        # default: 5e-3 # best: 1e-3
    "weight_decay"   : 0.0,         # NOTE: 1e-4 tested but kills learning (focal gradients too small vs L2 penalty);
                                    #   architecture already regularised via HGT dropout + augmentation dropout
    "max_epochs"     : 300,         # was 500; model never improves past epoch ~50 so 300 is sufficient
    "patience"       : 100,          # was 300 (wasteful); 75 gives headroom past epoch-50 peak without burning GPU
    "patient_chunk_size": 16384,    # patients per chunk in eval; 16384 for RTX 5090 32GB (was 4096 for T4 15GB)
    "use_gradient_checkpointing": False,  # False on RTX 5090 32GB: saves 2nd GNN forward pass per epoch;
                                          # set True only for GPUs < 16GB with FAERS_ALL
    "use_two_pass"   : False,       # False = single-pass training (needs ~4-6GB GPU, fastest);
                                    # True  = two-pass (OOM-safe for small GPUs, ~2× slower)
    "use_lr_scheduler": True,       # ReduceLROnPlateau on val_AUC: halves LR after 20 stagnant epochs
    # NEW PARAMETER FOR MODEL_V2
    "edge_drop_prob" : 0.1,
    "projection_dim" : 256,
    "projection_dropout" : 0.1,
    "similarity_threshold": 0.3,  # Jaccard similarity threshold for SupCon positive pairs (30% overlap)
    "alpha_range"    : [0.0, 0.01, 0.05, 0.1, 0.3],   # finer search near 0 where val evidence points
    "tau_range"      : [0.01, 0.05, 0.1],              # 15 combinations total (was 30)
    # ── Per-ADR positive weighting (Option C / REC-12) ───────────────────────
    "use_pos_weight" : True,    # weight rare ADRs more heavily in Focal Loss
    "pos_weight_cap" : 50.0,    # clamp max weight (avoids extreme values for ADRs
                                #   with only 1–2 positives in training)
    "alpha"          : 0.1,   # re-enabled for FAERS_ALL (200k patients vs 10k TB);
                               # with large N, InfoNCE contrastive signal is beneficial
    # alpha=0.0 was best for FAERS_TB (10k patients, 50 ADRs) — contrastive overfitting.
    # FAERS_ALL (200k patients) has enough diversity for InfoNCE to help.
    # Use grid search to tune over alpha_range above.
    "tau"            : 0.05,
    # InfoNCE configuration (per paper Equation 10)
    "use_sampled_negatives": False,   # False = exact paper formula (may OOM), True = memory-efficient
    "max_negatives"        : None,   # Maximum negative samples when use_sampled_negatives=True
    # Memory optimization settings
    "use_amp"              : True,    # Use automatic mixed precision (FP16) training
    "accumulation_steps"   : 1        # No accumulation needed on RTX 5090 32GB (was 4 for T4 15GB)
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

# ── Per-dataset MODEL overrides ───────────────────────────────────────────────
# Each entry overrides only the keys that differ from the base MODEL dict above.
# Applied at the bottom of this file so MODEL always reflects the active dataset.
#
# Rationale:
#   FAERS_TB  (~10k patients, ~50 ADRs): small graph → small model to avoid
#             overfitting; alpha=0.05 gives mild InfoNCE regularisation.
#   FAERS_ALL (~200k patients, 582 ADRs): large graph → paper-scale model;
#             single-pass training on RTX 5090; alpha=0.1 with large N.
_MODEL_OVERRIDES = {
    "FAERS_TB": {
        "embedding_dim"             : 128,   # confirmed best for ~10k patients
        "num_hgt_layers"            : 2,     # deeper overfits small graph
        "num_heads"                 : 2,
        "projection_dim"            : 128,
        "patient_chunk_size"        : 4096,
        "use_gradient_checkpointing": False, # small graph fits in memory
        "use_two_pass"              : False,
        "accumulation_steps"        : 1,
        "alpha"                     : 0.05,  # mild InfoNCE; 0.0 also fine
        "alpha_range"               : [0.0, 0.01, 0.05, 0.1],
    },
    "FAERS_TB_DRUGS": {
        "embedding_dim"             : 128,
        "num_hgt_layers"            : 2,
        "num_heads"                 : 2,
        "projection_dim"            : 128,
        "patient_chunk_size"        : 4096,
        "use_gradient_checkpointing": False,
        "use_two_pass"              : False,
        "accumulation_steps"        : 1,
        "alpha"                     : 0.05,
        "alpha_range"               : [0.0, 0.01, 0.05, 0.1],
    },
    "FAERS_ALL": {
        # Already the base MODEL defaults — no override needed.
        # Listed here for documentation clarity only.
        "embedding_dim"             : 256,
        "num_hgt_layers"            : 3,
        "num_heads"                 : 4,
        "projection_dim"            : 256,
        "patient_chunk_size"        : 16384,
        "use_gradient_checkpointing": False,
        "use_two_pass"              : False,
        "accumulation_steps"        : 1,
        "alpha"                     : 0.1,
    },
}

# Apply overrides for the active dataset
if DATASET_NAME in _MODEL_OVERRIDES:
    MODEL.update(_MODEL_OVERRIDES[DATASET_NAME])


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
    print(f"Model dim:     {MODEL['embedding_dim']}  "
          f"layers={MODEL['num_hgt_layers']}  heads={MODEL['num_heads']}  "
          f"alpha={MODEL['alpha']}")
    print("=" * 70)

if __name__ == "__main__":
    print_config()
