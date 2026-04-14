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
DATA_PATH = os.path.join(_PROJECT_ROOT, f"data/{DATASET_NAME}_2018")
PREPROCESSED_PATH = os.path.join(_PROJECT_ROOT, f"data/preprocessed_{DATASET_NAME.lower()}_2018")
OUTPUT_PATH = os.path.join(_PROJECT_ROOT, f"output_{DATASET_NAME.lower()}_2018")

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
    "primaryid"    : "primaryid",
    "age"          : "AGE",
    "gender"       : "Gender",
    "country_code" : "COUNTRY_CODE",
    "period"       : "Period",
    # DRUGS_STANDARDIZED
    "rxaui"        : "RXAUI",
    "drug_name"    : "DRUG",
    "drugbank_id"  : "DrugBank_ID",
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
}

# ── Quality-control settings ──────────────────────────────────────────────
# Note: For TB datasets, data is already filtered by TB criteria
QC = {
    "country_filter"  : "US",
    "reporter_filter" : "HP",
    # "min_frequency"   : 100,    # for FAERS_ALL
    "min_frequency" : 3       # for FAERS_TB and FAERS_TB_DRUGS (already filtered, so can use lower threshold)
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
    "top_n_adrs"    : None,      # e.g. 50 → keep top-50 most frequent ADRs
    "adr_list_file" : None       # e.g. "data/preprocessed_faers_tb_2018/top_50_adrs.txt"
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
    "edge_weight_mode"  : "log",    # "log" | "raw" | "none"
    "ddi_threshold"     : 3,
    "iii_threshold"     : 3,
}

# ── Model hyperparameters ─────────────────────────────────────────────────
MODEL = {
    "embedding_dim"  : 64,          # default: 256  # best: 128
    "num_hgt_layers" : 2,           # default: 3    # best: 4
    "num_heads"      : 2,           # default: 4    # best: 8
    "dropout"        : 0.1,         # default: 0.3  # best: 0.1         
    "focal_gamma"    : 1.0,         # default: 2.0  # best: 1.0
    "batch_size"     : 512,         # default: 512
    "lr"             : 1e-3,        # default: 5e-3 # best: 1e-3
    "max_epochs"     : 300,
    "patience"       : 100,
    
    # NEW PARAMETER FOR MODEL_V2
    "edge_drop_prob" : 0.1,
    "projection_dim" : 128,
    "projection_dropout" : 0.1,
    "similarity_threshold": 0.3,  # Jaccard similarity threshold for SupCon positive pairs (30% overlap)
            
    "alpha_range"    : [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
    "tau_range"      : [0.01, 0.05, 0.1, 0.2, 0.5],
    "alpha"          : 0.0,   # default; overridden by grid search
    "tau"            : 0.05,
    # InfoNCE configuration (per paper Equation 10)
    "use_sampled_negatives": False,   # False = exact paper formula (may OOM), True = memory-efficient
    "max_negatives"        : None,   # Maximum negative samples when use_sampled_negatives=True
    # Memory optimization settings for T4 GPU (15GB)
    "use_amp"              : True,    # Use automatic mixed precision (FP16) training
    "accumulation_steps"   : 4,       # Gradient accumulation steps (increase if still OOM)
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
        "description": "Full FAERS dataset (2018-2025)",
        "source": "data/standardized_faers_2018_2025",
    },
    "FAERS_TB": {
        "description": "Tuberculosis-related cases",
        "source": "data/standardized_faers_2018_2025 (filtered by FAERS_TB_PRIMARYID_SET)",
    },
    "FAERS_TB_DRUGS": {
        "description": "TB cases with TB-specific drugs",
        "source": "data/standardized_faers_2018_2025 (filtered by FAERS_TB_DRUGS_PRIMARYID_SET)",
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
