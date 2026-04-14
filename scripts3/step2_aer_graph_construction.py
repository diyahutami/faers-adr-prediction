"""
step2_aer_graph_construction.py
================================
Step 3 of the PreciseADR reproduction pipeline - scripts3 version.

SCRIPTS3 UPDATE: Multi-dataset support for FAERS_ALL, FAERS_TB, FAERS_TB_DRUGS
  - Reads dataset from environment variable DATASET_NAME or config.py
  - Data paths automatically adjusted based on selected dataset
  - Supports all three dataset variants with same graph construction

Constructs Heterogeneous Adverse Event Report (AER) Graphs for:
  • XXX (full dataset)
  • XXX-Gender
  • XXX-Age

For each variant, three graphs are built and saved:
  • train_graph.pt   – includes Patient-Experiences-ADR edges (training labels)
  • val_graph.pt     – no ADR edges for val patients (inference time)
  • test_graph.pt    – no ADR edges for test patients (inference time)

Node types  : Patient (P), Drug (D), Disease (I), ADR (A)
Edge types  : patient_takes_drug, patient_has_disease,
              patient_experiences_adr (train only)

Node features:
  Patient : BOW vector [age_bin_onehot(3), gender_onehot(2),
                        num_drugs_normed, num_diseases_normed] → d=256
  Drug    : one-hot over DrugBank_ID index                     → d=256
  Disease : one-hot over MEDDRA_CODE index                     → d=256
  ADR     : one-hot over MEDDRA_CODE index                     → d=256
  All projected to d=256 via a linear layer inside the model.

Edge weights (Drug-ADR):
  w = log(ROR + 1)  from PROPORTIONATE_ANALYSIS table

Usage
-----
    # Using default dataset from config.py
    python step2_aer_graph_construction.py [--variant xxx|xxx_gender|xxx_age]
    
    # Selecting specific dataset
    DATASET_NAME=FAERS_TB python step2_aer_graph_construction.py
    DATASET_NAME=FAERS_TB_DRUGS python step2_aer_graph_construction.py
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_PATH, PREPROCESSED_PATH, OUTPUT_PATH, GRAPH_PATH, FAERS_SEP, COL,
    AGE_BINS, AGE_LABELS, MODEL, ADR_FILTER, GRAPH, SIGNAL_EDGES, KG_EDGES,
    DATASET_NAME, print_config,
)

EMB_DIM = MODEL["embedding_dim"]

# ── Utility ──────────────────────────────────────────────────────────────────

def _load_filtered(name: str, preprocessed_path: str) -> pd.DataFrame:
    path = os.path.join(preprocessed_path, f"{name.upper()}_FILTERED.csv")
    return pd.read_csv(path, sep=FAERS_SEP, low_memory=False)


def _load_splits(preprocessed_path: str) -> dict:
    with open(os.path.join(preprocessed_path, "splits.json")) as f:
        raw = json.load(f)
    # Convert lists back to sets
    return {v: {s: set(ids) for s, ids in splits.items()}
            for v, splits in raw.items()}


def _load_prop_analysis(preprocessed_path: str) -> pd.DataFrame | None:
    path = os.path.join(preprocessed_path, "PROPORTIONATE_ANALYSIS.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, sep=FAERS_SEP, low_memory=False)
        df.columns = df.columns.str.upper()
        return df
    return None


def _age_bin(age: float) -> int:
    """Return 0/1/2 for Youth/Adult/Elderly."""
    if age < 18:  return 0
    if age < 65:  return 1
    return 2


def _patient_feature_vector(row: pd.Series,
                             num_drugs: int,
                             num_diseases: int,
                             max_drugs: float,
                             max_diseases: float) -> np.ndarray:
    """
    BOW patient feature vector:
      [age_youth, age_adult, age_elderly,  (3-dim one-hot)
       gender_male, gender_female,         (2-dim one-hot)
       num_drugs_normed,                   (scalar)
       num_diseases_normed]                (scalar)
    → length 7
    """
    age_onehot  = np.zeros(3)
    age_bin     = _age_bin(float(row.get(COL["age"], 40)))
    age_onehot[age_bin] = 1.0

    gender_onehot = np.zeros(2)
    g = str(row.get(COL["gender"], "")).upper()
    if g == "M": gender_onehot[0] = 1.0
    elif g == "F": gender_onehot[1] = 1.0

    nd = float(num_drugs)   / max_drugs    if max_drugs    > 0 else 0.0
    ni = float(num_diseases) / max_diseases if max_diseases > 0 else 0.0

    return np.concatenate([age_onehot, gender_onehot, [nd, ni]])  # shape (7,)


# ── Core graph builder ────────────────────────────────────────────────────────

class AERGraphBuilder:

    def __init__(self, data_path: str = DATA_PATH, 
                 preprocessed_path: str = PREPROCESSED_PATH,
                 output_path: str = OUTPUT_PATH,
                 graph_path: str = GRAPH_PATH):
        self.data_path   = data_path
        self.preprocessed_path = preprocessed_path
        self.output_path = output_path
        self.graph_path  = graph_path

        # Loaded tables (shared across all variants)
        self.demo: pd.DataFrame = None
        self.drugs: pd.DataFrame = None
        self.ar: pd.DataFrame = None          # adverse reactions
        self.di: pd.DataFrame = None          # drug indications
        self.splits: dict = None
        self.prop: pd.DataFrame = None        # proportionate analysis

        # Vocabulary maps (built from training set; shared)
        self.drug_vocab:    dict = {}   # DrugBank_ID → int
        self.disease_vocab: dict = {}   # MEDDRA_CODE  → int  (diseases)
        self.adr_vocab:     dict = {}   # MEDDRA_CODE  → int  (ADRs)
        self.target_vocab:  dict = {}   # UniProt target_id → int (KG Step 1)

        # Pair-level pharmacovigilance signal lookup for direct drug→ADR edges
        # (rxcui_str, pt_code_str) → np.ndarray([log_ror, ic, ror_lb_flag, log_a])
        self.prop_edge_lookup: dict = {}

        # Knowledge graph data (loaded once from knowledge_graph.pkl)
        self.kg_data:        dict = {}  # {edge_type → list of (src_id, dst_id)}
        self.umls_to_meddra: dict = {}  # UMLS_CUI → MedDRA_code (str)

        # KGIA disease relation data (loaded from biomedical_KG.txt)
        self.kgia_data:      dict = {}  # {"comorbidity": [...], "disease_risk": [...]}

        # DrugCentral drug-target data (loaded from drug.target.interaction.tsv)
        self.dc_data:        list = []  # list of (drugbank_id, "PROTEIN_<UniProt>") tuples

        # OnSIDES drug-label ADE data (loaded from high_confidence.csv)
        self.onsides_data:   list = []  # list of (rxcui_str, meddra_code_str) pairs
    # ── load ─────────────────────────────────────────────────────────────────

    def load(self):
        print("Loading filtered tables …")
        self.demo  = _load_filtered("demographics",      self.preprocessed_path)
        self.drugs = _load_filtered("drugs_standardized",self.preprocessed_path)
        self.ar    = _load_filtered("adverse_reactions",  self.preprocessed_path)
        self.di    = _load_filtered("drug_indications",   self.preprocessed_path)
        self.splits = _load_splits(self.preprocessed_path)
        self.prop   = _load_prop_analysis(self.preprocessed_path)
        print("  ✓ Tables loaded")
        self._load_kg_data()
        self._load_kgia_data()
        self._load_drugcentral_data()
        self._load_onsides_data()

    # ── vocabulary ────────────────────────────────────────────────────────────

    def _build_vocab(self, train_pids: set):
        """
        Build entity vocabularies from the training set only.

        ADR label space can be restricted via ADR_FILTER in config.py:
          - adr_list_file: path to a .txt file (one ADR name per line) from
                           analyze_adr_frequency.py --save-list
          - top_n_adrs:    integer N → keep only the N most-frequent ADRs
                           computed over the training set
          If both are set, adr_list_file takes priority.
          If neither is set, all ADRs in the training set are kept.
        """
        # Drug vocabulary: prefer DrugBank_ID; fallback to RXCUI
        drug_col = COL["drugbank_id"] if COL["drugbank_id"] in self.drugs.columns else COL["rxcui"]
        train_drugs = self.drugs[self.drugs[COL["primaryid"]].isin(train_pids)]
        unique_drugs = sorted(train_drugs[drug_col].dropna().unique())
        self.drug_vocab = {d: i for i, d in enumerate(unique_drugs)}

        # Disease vocabulary: prefer MEDDRA_CODE; fallback to DRUG_INDICATION text
        di_col = COL["meddra_code"] if COL["meddra_code"] in self.di.columns else COL["indication"]
        train_di = self.di[self.di[COL["primaryid"]].isin(train_pids)]
        unique_dis = sorted(train_di[di_col].dropna().astype(str).unique())
        self.disease_vocab = {d: i for i, d in enumerate(unique_dis)}

        # ADR vocabulary: prefer MEDDRA_CODE; fallback to ADVERSE_EVENT text
        ar_col = COL["meddra_code"] if COL["meddra_code"] in self.ar.columns else COL["adverse_event"]
        train_ar = self.ar[self.ar[COL["primaryid"]].isin(train_pids)]

        # ── ADR label filtering ────────────────────────────────────────────
        adr_list_file = ADR_FILTER.get("adr_list_file")
        top_n_adrs    = ADR_FILTER.get("top_n_adrs")

        if adr_list_file and os.path.exists(adr_list_file):
            # Load explicit ADR list from file (one name per line)
            with open(adr_list_file) as f:
                allowed_adrs = set(line.strip() for line in f if line.strip())
            # The file uses ADVERSE_EVENT text names; filter the ar_col accordingly.
            # If ar_col is MEDDRA_CODE we need to map names → codes first.
            if ar_col == COL["adverse_event"]:
                train_ar = train_ar[train_ar[ar_col].isin(allowed_adrs)]
            else:
                # ar_col is MEDDRA_CODE: resolve names to codes via ADVERSE_EVENT column
                name_to_code = (
                    train_ar[[COL["adverse_event"], ar_col]]
                    .dropna()
                    .drop_duplicates()
                    .set_index(COL["adverse_event"])[ar_col]
                    .to_dict()
                )
                allowed_codes = {name_to_code[n] for n in allowed_adrs if n in name_to_code}
                train_ar = train_ar[train_ar[ar_col].isin(allowed_codes)]
            print(f"  ADR filter: file '{os.path.basename(adr_list_file)}' "
                  f"→ {train_ar[ar_col].nunique():,} ADRs retained")

        elif top_n_adrs is not None:
            # Keep only the top-N most frequent ADRs by record count in training set
            freq = train_ar[ar_col].value_counts()
            top_codes = set(freq.head(int(top_n_adrs)).index)
            train_ar = train_ar[train_ar[ar_col].isin(top_codes)]
            print(f"  ADR filter: top_n_adrs={top_n_adrs} "
                  f"→ {train_ar[ar_col].nunique():,} ADRs retained "
                  f"(min freq={int(freq.iloc[top_n_adrs - 1]) if len(freq) >= top_n_adrs else 0})")

        else:
            print(f"  ADR filter: none (all {train_ar[ar_col].nunique():,} ADRs kept)")
        # ── end ADR filtering ──────────────────────────────────────────────

        unique_adrs = sorted(train_ar[ar_col].dropna().astype(str).unique())
        self.adr_vocab = {a: i for i, a in enumerate(unique_adrs)}

        print(f"  Vocabulary: {len(self.drug_vocab):,} drugs | "
              f"{len(self.disease_vocab):,} diseases | "
              f"{len(self.adr_vocab):,} ADRs")

    # ── edge-weight lookup ────────────────────────────────────────────────────

    # def _build_ror_map(self) -> dict:
    #     """Return dict (drug_id, adr_id) → log(ROR+1) from PROPORTIONATE_ANALYSIS."""
    #     if self.prop is None:
    #         return {}
    #     ror_map = {}
    #     drug_col = COL["rxaui"]
    #     ar_col   = COL["meddra_code"] if COL["meddra_code"] in self.prop.columns else "PT_CODE"
    #     for _, row in self.prop.iterrows():
    #         d = str(row.get(drug_col, ""))
    #         a = str(row.get(ar_col, ""))
    #         ror = row.get(COL["ror"], 0)
    #         if pd.notna(ror) and ror > 0:
    #             ror_map[(d, a)] = float(np.log(float(ror) + 1))
    #     return ror_map

    # proportionate analysis node features
    def _build_drugvocab_to_rxcui_map(self) -> dict:
        """
        Build a mapping from drug-vocabulary keys (DrugBank_ID or RXCUI) back to
        RXCUI strings, so prop features (keyed by RXCUI) can be looked up for any
        drug in the vocabulary

        Returns:
            dict: vocab_key_str → RXCUI string
        """
        drug_col = self._drug_id_col() # DrugBank_ID if present, else RXCUI
        if drug_col == COL["rxcui"]:
            # vocab is already keyed by RXCUI - identity mapping
            return {k: k for k in self.drug_vocab}
        # vocab is keyed by DrugBank_ID - need to join back to RXCUI
        mapping = (
            self.drugs[[COL["rxcui"], drug_col]]
            .dropna()
            .drop_duplicates(subset=[drug_col])
            .set_index(drug_col)[COL["rxcui"]]
            .astype(str)
            .to_dict()
        )
        return mapping

    # ── pharmacovigilance signal edge lookup ──────────────────────────────────

    def _build_prop_edge_lookup(self) -> None:
        """
        Build pair-level pharmacovigilance signal lookup from PROPORTIONATE_ANALYSIS.
        Stores only statistically significant pairs (A >= min_obs AND IC025 > threshold).

        Populates self.prop_edge_lookup:
            (rxcui_str, pt_code_str) → np.ndarray shape (4,)
                [0] log_ror     = log(clip(ROR, 0.01, ror_clip) + 1)
                [1] ic          = clip(IC, -ic_clip, ic_clip)
                [2] ror_lb_flag = float(ROR_LB > 1)
                [3] log_a       = log(A + 1)
        """
        self.prop_edge_lookup = {}

        if self.prop is None or not SIGNAL_EDGES.get("enabled", True):
            return

        min_obs   = int(SIGNAL_EDGES.get("min_obs_count", 3))
        ic025_thr = float(SIGNAL_EDGES.get("ic025_threshold", 0.0))
        ror_clip  = float(SIGNAL_EDGES.get("ror_clip", 50.0))
        ic_clip   = float(SIGNAL_EDGES.get("ic_clip", 5.0))

        rxcui_col  = COL["rxcui"]
        pt_col     = COL["pt_code"]
        obs_col    = COL["obs_count"]
        ror_col    = COL["ror"]
        ror_lb_col = COL["ror_lb"]
        ic_col     = COL["ic"]
        ic025_col  = COL["ic025"]

        prop = self.prop.copy()
        prop[rxcui_col] = prop[rxcui_col].astype(str).str.strip()
        prop[pt_col]    = prop[pt_col].astype(str).str.strip()

        # Keep only statistically significant pairs
        mask = (
            prop[rxcui_col].ne("") &
            prop[pt_col].ne("") &
            prop[obs_col].notna() &
            prop[ror_col].notna() &
            prop[ror_lb_col].notna() &
            prop[ic_col].notna() &
            prop[ic025_col].notna() &
            (prop[obs_col].astype(float) >= min_obs) &
            (prop[ic025_col].astype(float) > ic025_thr)
        )
        prop = prop[mask].copy()

        print(f" Signal edges: {len(prop):,} significant pairs "
              f"(A>={min_obs}, IC025>{ic025_thr}) from {len(self.prop):,} total")

        if prop.empty:
            print(" No significant pairs found — signal edges disabled")
            return

        # Compute features (vectorized)
        prop["_log_ror"]     = np.log(np.clip(prop[ror_col].astype(float), 0.01, ror_clip) + 1.0)
        prop["_ic"]          = np.clip(prop[ic_col].astype(float), -ic_clip, ic_clip)
        prop["_ror_lb_flag"] = (prop[ror_lb_col].astype(float) > 1.0).astype(np.float32)
        prop["_log_a"]       = np.log(prop[obs_col].astype(float) + 1.0)

        for _, row in prop.iterrows():
            key = (str(row[rxcui_col]), str(row[pt_col]))
            self.prop_edge_lookup[key] = np.array([
                float(row["_log_ror"]),
                float(row["_ic"]),
                float(row["_ror_lb_flag"]),
                float(row["_log_a"]),
            ], dtype=np.float32)

        print(f" Prop edge lookup: {len(self.prop_edge_lookup):,} entries built")

    # ── Knowledge Graph loading & integration ─────────────────────────────────

    def _load_kg_data(self) -> None:
        """
        Load knowledge_graph.pkl produced by ADR_Prediction/scripts_improved/.
        Builds:
          self.kg_data        — {edge_type: list of (src_id, dst_id)}
          self.umls_to_meddra — {UMLS_CUI_str: MedDRA_code_str}

        Called once in load(); skipped if KG_EDGES is disabled or file missing.
        """
        if not KG_EDGES.get("enabled", False):
            return

        kg_path = KG_EDGES.get("kg_data_path", "")
        if not os.path.exists(kg_path):
            print(f"  ⚠ KG data not found at {kg_path} — KG edges disabled")
            return

        import pickle
        print("  Loading external knowledge graph …")
        with open(kg_path, "rb") as f:
            raw = pickle.load(f)
        self.kg_data = raw.get("edge_types", {})
        print(f"    ✓ KG loaded: {sum(len(v) for v in self.kg_data.values()):,} total edges "
              f"across {len(self.kg_data)} edge types")

        # Build UMLS CUI → MedDRA code mapping (PT term type only)
        meddra_path = KG_EDGES.get("meddra_path", "")
        if os.path.exists(meddra_path):
            meddra_df = pd.read_csv(
                meddra_path, sep="\t", header=None,
                names=["umls_cui", "term_type", "meddra_code", "name"],
                dtype=str
            )
            pt_df = (meddra_df[meddra_df["term_type"] == "PT"]
                     .drop_duplicates(subset=["umls_cui"])
                     .set_index("umls_cui")["meddra_code"])
            self.umls_to_meddra = pt_df.to_dict()
            print(f"    ✓ MedDRA mapping: {len(self.umls_to_meddra):,} UMLS→MedDRA entries")

    def _build_kg_target_vocab(self) -> None:
        """
        Build target_vocab by merging targets from all enabled sources:
          - DrugBank (PROTEIN_<UniProt> format from knowledge_graph.pkl)
          - DrugCentral (PROTEIN_<UniProt> format; PROTEIN_ prefix added on load)
        Proteins shared between sources map to the SAME target node.
        Must be called AFTER _build_vocab() and all _load_*() methods.
        """
        self.target_vocab = {}
        drug_set = set(self.drug_vocab.keys())
        reachable: set = set()

        # DrugBank drug-target edges
        n_db = 0
        if KG_EDGES.get("use_drug_target", False) and self.kg_data:
            db_targets = {dst for src, dst in self.kg_data.get("drug-target", [])
                          if src in drug_set}
            reachable |= db_targets
            n_db = len(db_targets)

        # DrugCentral drug-target edges (PROTEIN_<UniProt> format)
        n_dc = 0
        if KG_EDGES.get("use_drugcentral_target", False) and self.dc_data:
            dc_targets = {t for d, t in self.dc_data if d in drug_set}
            reachable |= dc_targets
            n_dc = len(dc_targets)

        if not reachable:
            return

        self.target_vocab = {t: i for i, t in enumerate(sorted(reachable))}
        shared = n_db + n_dc - len(reachable) if (n_db and n_dc) else 0
        print(f"  Target vocab: {len(self.target_vocab):,} proteins "
              f"(DrugBank={n_db:,}, DrugCentral={n_dc:,}"
              f"{f', shared={shared:,}' if shared else ''})")

    def _target_features(self) -> torch.Tensor:
        """One-hot identity matrix over target (UniProt protein) index."""
        return torch.eye(len(self.target_vocab), dtype=torch.float)

    def _build_kg_ddi_edges(self) -> tuple:
        """
        Step 2 — DrugBank pharmacological DDI edges.
        Edge type: (drug, db_interacts, drug)  [separate from FAERS co-occurrence]

        Filters to both endpoints in drug_vocab.
        Returns (edge_index [2, E], None).
        """
        if not self.kg_data or not KG_EDGES.get("use_drugbank_ddi", False):
            return torch.zeros((2, 0), dtype=torch.long), None

        drug_set = set(self.drug_vocab.keys())
        src_list, dst_list = [], []
        for src, dst in self.kg_data.get("drug-drug", []):
            si = self.drug_vocab.get(src)
            di = self.drug_vocab.get(dst)
            if si is not None and di is not None:
                src_list.append(si);  dst_list.append(di)
                src_list.append(di);  dst_list.append(si)   # bidirectional

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    DrugBank DDI edges: {edge_index.shape[1]:,} "
              f"(from {len(set(src_list)):,} drugs)")
        return edge_index, None

    def _build_kg_drug_target_edges(self) -> tuple:
        """
        Step 1 — DrugBank drug-target edges.
        Edge type: (drug, has_target, target) + reverse (target, targeted_by, drug)

        Requires self.target_vocab to be built first via _build_kg_target_vocab().
        Returns (edge_index [2, E], None).
        """
        if not self.kg_data or not self.target_vocab:
            return torch.zeros((2, 0), dtype=torch.long), None

        src_list, dst_list = [], []
        for src, dst in self.kg_data.get("drug-target", []):
            di = self.drug_vocab.get(src)
            ti = self.target_vocab.get(dst)
            if di is not None and ti is not None:
                src_list.append(di)
                dst_list.append(ti)

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    Drug-target edges: {edge_index.shape[1]:,} "
              f"({edge_index[0].unique().shape[0]:,} drugs → "
              f"{edge_index[1].unique().shape[0]:,} targets)")
        return edge_index, None

    def _build_kg_sider_adr_edges(self) -> tuple:
        """
        Step 3 — SIDER population-level drug-ADR prior knowledge.
        Edge type: (drug, kg_has_adr, adr) + reverse (adr, kg_adr_of, drug)

        Mapping:  DrugBank_ID (KG) → drug_vocab
                  'ADR_<UMLS_CUI>' → strip prefix → umls_to_meddra → MedDRA code str → adr_vocab

        Returns (edge_index [2, E], None).
        """
        if not self.kg_data or not self.umls_to_meddra or not KG_EDGES.get("use_sider_adr", False):
            return torch.zeros((2, 0), dtype=torch.long), None

        src_list, dst_list = [], []
        seen = set()   # deduplicate (drug_idx, adr_idx) pairs

        for src, dst in self.kg_data.get("drug-adr", []):
            di = self.drug_vocab.get(src)
            if di is None:
                continue
            # dst is like 'ADR_C0000729'
            if not dst.startswith("ADR_"):
                continue
            umls_cui = dst[4:]   # strip 'ADR_' prefix
            meddra_code = self.umls_to_meddra.get(umls_cui)
            if meddra_code is None:
                continue
            ai = self.adr_vocab.get(str(meddra_code))
            if ai is None:
                continue
            pair = (di, ai)
            if pair not in seen:
                seen.add(pair)
                src_list.append(di)
                dst_list.append(ai)

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    SIDER drug-ADR edges: {edge_index.shape[1]:,} unique pairs "
              f"({edge_index[0].unique().shape[0]:,} drugs → "
              f"{edge_index[1].unique().shape[0]:,} ADRs)")
        return edge_index, None

    def _load_kgia_data(self) -> None:
        """
        Load KGIA biomedical_KG.txt and extract disease-disease relation triples.

        Populates self.kgia_data:
            "comorbidity"  — list of (umls_cui_src, umls_cui_dst) from COMORBIDITY rows
                             KGIA already provides both directions; we keep all rows.
            "disease_risk" — list of (umls_cui_src, umls_cui_dst) from DISEASE_RISK rows
                             Directed: src has risk factor dst (or src predisposes to dst).

        Skipped if neither use_kgia_comorbidity nor use_kgia_risk is enabled.
        """
        self.kgia_data = {}
        need = (KG_EDGES.get("use_kgia_comorbidity", False) or
                KG_EDGES.get("use_kgia_risk", False))
        if not need or not KG_EDGES.get("enabled", False):
            return

        kgia_path = KG_EDGES.get("kgia_data_path", "")
        if not os.path.exists(kgia_path):
            print(f"  ⚠ KGIA data not found at {kgia_path} — KGIA edges disabled")
            return

        comorbidity:  list = []
        disease_risk: list = []

        print("  Loading KGIA biomedical knowledge graph …")
        with open(kgia_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                src, rel, dst = parts
                if rel == "COMORBIDITY":
                    comorbidity.append((src, dst))
                elif rel == "DISEASE_RISK":
                    disease_risk.append((src, dst))

        self.kgia_data["comorbidity"]  = comorbidity
        self.kgia_data["disease_risk"] = disease_risk
        print(f"    ✓ KGIA: {len(comorbidity):,} comorbidity pairs, "
              f"{len(disease_risk):,} disease-risk pairs")

    def _build_kgia_comorbidity_edges(self) -> tuple:
        """
        Step 4 — KGIA disease comorbidity edges.
        Edge type: (disease, kg_comorbid, disease)

        Maps UMLS CUI → MedDRA code → disease_vocab for both endpoints.
        Deduplicates (src_idx, dst_idx) pairs.
        Returns (edge_index [2, E], None).
        """
        if not self.kgia_data or not KG_EDGES.get("use_kgia_comorbidity", False):
            return torch.zeros((2, 0), dtype=torch.long), None

        src_list, dst_list = [], []
        seen: set = set()

        for umls_src, umls_dst in self.kgia_data.get("comorbidity", []):
            mc_src = self.umls_to_meddra.get(umls_src)
            mc_dst = self.umls_to_meddra.get(umls_dst)
            if mc_src is None or mc_dst is None:
                continue
            si = self.disease_vocab.get(str(mc_src))
            di = self.disease_vocab.get(str(mc_dst))
            if si is None or di is None or si == di:
                continue
            pair = (si, di)
            if pair not in seen:
                seen.add(pair)
                src_list.append(si)
                dst_list.append(di)

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    KGIA comorbidity edges: {edge_index.shape[1]:,} "
              f"({edge_index[0].unique().shape[0]:,} unique disease nodes)")
        return edge_index, None

    def _build_kgia_risk_edges(self) -> tuple:
        """
        Step 5 — KGIA disease-risk factor edges.
        Edge types: (disease, kg_has_risk, disease) +
                    (disease, kg_risk_of,  disease)  [reverse]

        Maps UMLS CUI → MedDRA code → disease_vocab for both endpoints.
        Returns (edge_index [2, E], None) for the forward direction;
        caller adds the reverse.
        """
        if not self.kgia_data or not KG_EDGES.get("use_kgia_risk", False):
            return torch.zeros((2, 0), dtype=torch.long), None

        src_list, dst_list = [], []
        seen: set = set()

        for umls_src, umls_dst in self.kgia_data.get("disease_risk", []):
            mc_src = self.umls_to_meddra.get(umls_src)
            mc_dst = self.umls_to_meddra.get(umls_dst)
            if mc_src is None or mc_dst is None:
                continue
            si = self.disease_vocab.get(str(mc_src))
            di = self.disease_vocab.get(str(mc_dst))
            if si is None or di is None or si == di:
                continue
            pair = (si, di)
            if pair not in seen:
                seen.add(pair)
                src_list.append(si)
                dst_list.append(di)

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    KGIA risk-factor edges: {edge_index.shape[1]:,} "
              f"({edge_index[0].unique().shape[0]:,} disease sources)")
        return edge_index, None

    def _load_drugcentral_data(self) -> None:
        """
        Load DrugCentral 2023 drug-target interactions (human only).
        Maps drug names → DrugBank IDs via two lookups (in priority order):
          1. drug_id_mapping.tsv  (DrugBank-curated cross-reference)
          2. FAERS drugs table    (drug name → DrugBank_ID already in self.drugs)
        Target IDs normalised to PROTEIN_<UniProt> format to match DrugBank KG.

        Populates self.dc_data: list of (drugbank_id, "PROTEIN_<UniProt>") tuples.
        Called once in load(); skipped if use_drugcentral_target is False.
        """
        self.dc_data = []
        if not KG_EDGES.get("use_drugcentral_target", False) or not KG_EDGES.get("enabled", False):
            return

        dc_path      = KG_EDGES.get("drugcentral_data_path", "")
        mapping_path = KG_EDGES.get("drugcentral_mapping_path", "")

        if not os.path.exists(dc_path):
            print(f"  ⚠ DrugCentral data not found at {dc_path} — DrugCentral edges disabled")
            return

        print("  Loading DrugCentral drug-target data …")

        # ── Build drug name → DrugBank ID lookup ─────────────────────────────
        name_to_db: dict = {}

        # Source 1: drug_id_mapping.tsv (broader cross-reference)
        if os.path.exists(mapping_path):
            mapping_df = pd.read_csv(mapping_path, sep="\t", dtype=str)
            valid = mapping_df.dropna(subset=["drugbankId", "name"])
            name_to_db = (valid.assign(name_lower=valid["name"].str.lower().str.strip())
                          .set_index("name_lower")["drugbankId"]
                          .to_dict())

        # Source 2: FAERS drugs table (higher priority — already vocab-matched)
        if self.drugs is not None:
            drug_col = self._drug_id_col()
            if drug_col == COL["drugbank_id"]:
                faers_map = (
                    self.drugs.dropna(subset=[COL["drugbank_id"], COL["drug_name"]])
                    .assign(_name=self.drugs[COL["drug_name"]].str.lower().str.strip())
                    .groupby("_name")[COL["drugbank_id"]].first()
                    .to_dict()
                )
                name_to_db.update(faers_map)   # FAERS overrides on conflict

        # ── Load DrugCentral TSV ──────────────────────────────────────────────
        dc = pd.read_csv(dc_path, sep="\t", dtype=str)
        dc.columns = dc.columns.str.strip('"')
        dc["DRUG_NAME"] = dc["DRUG_NAME"].str.strip('"').str.lower().str.strip()
        dc["ACCESSION"] = dc["ACCESSION"].str.strip('"').str.strip()

        # Filter: Homo sapiens only, valid UniProt accession
        dc = dc[dc["ORGANISM"].str.strip('"').str.lower().str.contains("homo sapiens", na=False)]
        dc = dc[dc["ACCESSION"].notna() & (dc["ACCESSION"] != "")]

        # Map drug names → DrugBank IDs
        dc["DRUGBANK_ID"] = dc["DRUG_NAME"].map(name_to_db)
        dc = dc.dropna(subset=["DRUGBANK_ID"]).copy()

        # Deduplicate and store with PROTEIN_ prefix (matches DrugBank KG format)
        pairs = (dc[["DRUGBANK_ID", "ACCESSION"]]
                 .drop_duplicates()
                 .assign(target_id="PROTEIN_" + dc["ACCESSION"]))
        self.dc_data = list(zip(pairs["DRUGBANK_ID"], pairs["target_id"]))

        n_drugs   = len({d for d, _ in self.dc_data})
        n_targets = len({t for _, t in self.dc_data})
        print(f"    ✓ DrugCentral: {len(self.dc_data):,} drug-target pairs "
              f"({n_drugs:,} drugs, {n_targets:,} unique targets)")

    def _build_drugcentral_target_edges(self) -> tuple:
        """
        Step 6 — DrugCentral drug-target edges.
        Edge type: (drug, dc_has_target, target) + reverse (target, dc_targeted_by, drug)

        Both DrugBank and DrugCentral targets share the same target_vocab
        (unified PROTEIN_<UniProt> namespace), so overlapping proteins are
        the same graph node — enabling cross-source message passing.

        Returns (edge_index [2, E], None).
        """
        if not self.dc_data or not KG_EDGES.get("use_drugcentral_target", False):
            return torch.zeros((2, 0), dtype=torch.long), None
        if not self.target_vocab:
            return torch.zeros((2, 0), dtype=torch.long), None

        src_list, dst_list = [], []
        seen: set = set()

        for db_id, target_id in self.dc_data:
            di = self.drug_vocab.get(db_id)
            ti = self.target_vocab.get(target_id)
            if di is None or ti is None:
                continue
            pair = (di, ti)
            if pair not in seen:
                seen.add(pair)
                src_list.append(di)
                dst_list.append(ti)

        if not src_list:
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    DrugCentral drug-target edges: {edge_index.shape[1]:,} "
              f"({edge_index[0].unique().shape[0]:,} drugs → "
              f"{edge_index[1].unique().shape[0]:,} targets)")
        return edge_index, None

    # ── OnSIDES ───────────────────────────────────────────────────────────────

    def _load_onsides_data(self) -> None:
        """
        Step 7 — Load OnSIDES high_confidence.csv drug-ADE pairs.

        Populates self.onsides_data:
            list of (rxcui_str, meddra_code_str) tuples
              rxcui_str       = RxNorm ingredient-level RXCUI (matches FAERS RXCUI)
              meddra_code_str = MedDRA PT numeric code (matches FAERS MEDDRA_CODE)

        Skipped unless use_onsides_adr is True in KG_EDGES.
        Source: high_confidence.csv (high-precision NLP-extracted drug-ADE pairs,
                663 pairs across 71 RxNorm ingredient RXCUIs and 180 MedDRA PTs).
        """
        self.onsides_data = []
        if not KG_EDGES.get("use_onsides_adr", False) or not KG_EDGES.get("enabled", False):
            return

        onsides_path = KG_EDGES.get("onsides_data_path", "")
        hc_file = os.path.join(onsides_path, "high_confidence.csv")
        if not os.path.exists(hc_file):
            print(f"  [OnSIDES] high_confidence.csv not found: {hc_file}")
            return

        hc = pd.read_csv(hc_file)
        n_drugs = hc["ingredient_id"].nunique()
        n_adrs  = hc["effect_meddra_id"].nunique()

        for _, row in hc.iterrows():
            try:
                rxcui  = str(int(row["ingredient_id"]))
                meddra = str(int(row["effect_meddra_id"]))
                self.onsides_data.append((rxcui, meddra))
            except (ValueError, TypeError):
                continue

        print(f"  [OnSIDES] Loaded {len(self.onsides_data):,} drug-ADE pairs "
              f"({n_drugs:,} ingredient RXCUIs, {n_adrs:,} MedDRA PTs)")

    def _build_onsides_adr_edges(self) -> tuple:
        """
        Step 7 — OnSIDES drug-label ADE prior knowledge.
        Edge type: (drug, onsides_has_adr, adr) + reverse (adr, onsides_adr_of, drug)

        Drug mapping strategy:
          - If drug_vocab is keyed by RXCUI: direct lookup (ONSIDES ingredient_id = RXCUI).
          - If drug_vocab is keyed by DrugBank_ID: build RXCUI→DrugBank_ID reverse map
            from the drugs table, then look up via DrugBank_ID.

        ADR mapping:
          - ONSIDES effect_meddra_id = MedDRA PT numeric code → adr_vocab key (str).

        Returns (edge_index [2, E], None).
        """
        if not self.onsides_data or not KG_EDGES.get("use_onsides_adr", False):
            return torch.zeros((2, 0), dtype=torch.long), None

        drug_col = self._drug_id_col()  # "RXCUI" or "DRUGBANK_ID"

        # Build RXCUI → drug_vocab_idx lookup
        if drug_col == COL["rxcui"]:
            # drug_vocab is keyed by RXCUI already — direct match
            rxcui_to_drug_idx = {str(k): v for k, v in self.drug_vocab.items()}
        else:
            # drug_vocab is keyed by DrugBank_ID — need RXCUI → DrugBank_ID → idx
            rxcui_to_dbid = (
                self.drugs[[COL["rxcui"], COL["drugbank_id"]]]
                .dropna()
                .drop_duplicates(subset=[COL["rxcui"]])
                .set_index(COL["rxcui"])[COL["drugbank_id"]]
                .astype(str)
                .to_dict()
            )
            rxcui_to_drug_idx = {}
            for rxcui_str, dbid in rxcui_to_dbid.items():
                idx = self.drug_vocab.get(dbid)
                if idx is not None:
                    rxcui_to_drug_idx[str(rxcui_str)] = idx

        src_list, dst_list = [], []
        seen: set = set()

        for rxcui, meddra in self.onsides_data:
            drug_idx = rxcui_to_drug_idx.get(rxcui)
            if drug_idx is None:
                continue
            adr_idx = self.adr_vocab.get(meddra)
            if adr_idx is None:
                continue
            pair = (drug_idx, adr_idx)
            if pair not in seen:
                seen.add(pair)
                src_list.append(drug_idx)
                dst_list.append(adr_idx)

        if not src_list:
            print("    [OnSIDES] 0 drug-ADE pairs matched vocab "
                  "(check RXCUI / MedDRA alignment between datasets)")
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        print(f"    OnSIDES drug-ADE edges: {edge_index.shape[1]:,} unique pairs "
              f"({edge_index[0].unique().shape[0]:,} drugs → "
              f"{edge_index[1].unique().shape[0]:,} ADRs)")
        return edge_index, None

    def _build_signal_edges(self) -> tuple:
        """
        Build (drug, signals_adr, adr) edges from self.prop_edge_lookup.

        Maps each (rxcui, pt_code) entry to (drug_node_idx, adr_node_idx) using
        the current drug_vocab and adr_vocab.

        Returns:
            (edge_index, edge_attr)
              edge_index : LongTensor shape (2, E)  — [drug_idx, adr_idx]
              edge_attr  : FloatTensor shape (E, 4) — [log_ror, ic, ror_lb_flag, log_a]
            Returns empty tensors if no matching pairs found.
        """
        if not self.prop_edge_lookup:
            return torch.zeros((2, 0), dtype=torch.long), None

        # ── RXCUI → drug node index ───────────────────────────────────────────
        vocab_to_rxcui = self._build_drugvocab_to_rxcui_map()  # vocab_key → rxcui
        rxcui_to_drug_idx: dict = {}
        for vocab_key, rxcui in vocab_to_rxcui.items():
            drug_idx = self.drug_vocab.get(vocab_key)
            if drug_idx is not None:
                rxcui_to_drug_idx[str(rxcui)] = drug_idx

        # ── PT_CODE → adr node index ──────────────────────────────────────────
        # adr_vocab may be keyed by MEDDRA_CODE (= PT_CODE) or ADVERSE_EVENT text
        pt_to_adr_idx: dict = {}
        adr_col = self._adr_id_col()

        if adr_col == COL["meddra_code"]:
            for vocab_key, adr_idx in self.adr_vocab.items():
                pt_to_adr_idx[str(vocab_key)] = adr_idx
        else:
            # Fallback: resolve ADVERSE_EVENT text → PT_CODE via prop table
            if (self.prop is not None
                    and COL["pt_code"] in self.prop.columns
                    and COL["adverse_event"] in self.prop.columns):
                name_to_pt = (
                    self.prop[[COL["adverse_event"], COL["pt_code"]]]
                    .dropna()
                    .drop_duplicates(subset=[COL["adverse_event"]])
                    .set_index(COL["adverse_event"])[COL["pt_code"]]
                    .astype(str)
                    .to_dict()
                )
                for vocab_key, adr_idx in self.adr_vocab.items():
                    pt = name_to_pt.get(str(vocab_key), "")
                    if pt:
                        pt_to_adr_idx[pt] = adr_idx

        # ── Build edge tensors ────────────────────────────────────────────────
        src_list:  list = []
        dst_list:  list = []
        feat_list: list = []

        for (rxcui, pt), feat in self.prop_edge_lookup.items():
            drug_idx = rxcui_to_drug_idx.get(rxcui)
            adr_idx  = pt_to_adr_idx.get(pt)
            if drug_idx is not None and adr_idx is not None:
                src_list.append(drug_idx)
                dst_list.append(adr_idx)
                feat_list.append(feat)

        if not src_list:
            print(" Signal edges: 0 drug-ADR pairs matched vocab "
                  "(check RXCUI / PT_CODE alignment between tables)")
            return torch.zeros((2, 0), dtype=torch.long), None

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(np.stack(feat_list), dtype=torch.float)

        print(f" Signal edges built: {edge_index.shape[1]:,} drug→ADR edges "
              f"(from {len(rxcui_to_drug_idx):,} mapped drugs, "
              f"{len(pt_to_adr_idx):,} mapped ADRs)")

        return edge_index, edge_attr

    # ── node features ─────────────────────────────────────────────────────────

    def _patient_features(self, pids: list) -> torch.Tensor:
        """
        Build patient feature matrix (N_patients × 7).
        Counts come from the full (train+val/test) tables so test patients
        can still have drug/disease counts.
        """
        pid_to_idx = {p: i for i, p in enumerate(pids)}
        demo_sub = self.demo[self.demo[COL["primaryid"]].isin(pids)].set_index(COL["primaryid"])

        n_drugs_per_pid = self.drugs[self.drugs[COL["primaryid"]].isin(pids)].groupby(
            COL["primaryid"])[COL["drug_name"]].count()
        n_dis_per_pid = self.di[self.di[COL["primaryid"]].isin(pids)].groupby(
            COL["primaryid"])[COL["indication"]].count()

        max_drugs = max(n_drugs_per_pid.max(), 1)
        max_dis   = max(n_dis_per_pid.max(), 1)

        feats = np.zeros((len(pids), 7), dtype=np.float32)
        for pid, idx in pid_to_idx.items():
            row      = demo_sub.loc[pid] if pid in demo_sub.index else {}
            nd       = n_drugs_per_pid.get(pid, 0)
            ni       = n_dis_per_pid.get(pid, 0)
            feats[idx] = _patient_feature_vector(row, nd, ni, max_drugs, max_dis)

        return torch.tensor(feats, dtype=torch.float)

    def _drug_features(self) -> torch.Tensor:
        """One-hot identity matrix over DrugBank_ID (or RXCUI) index."""
        return torch.eye(len(self.drug_vocab), dtype=torch.float)

    def _disease_features(self) -> torch.Tensor:
        return torch.eye(len(self.disease_vocab), dtype=torch.float)

    def _adr_features(self) -> torch.Tensor:
        """One-hot identity matrix over ADR (MEDDRA_CODE or text) index."""
        return torch.eye(len(self.adr_vocab), dtype=torch.float)


    # ── edge building ─────────────────────────────────────────────────────────

    def _drug_id_col(self):
        return COL["drugbank_id"] if COL["drugbank_id"] in self.drugs.columns else COL["rxcui"]

    def _disease_id_col(self):
        return COL["meddra_code"] if COL["meddra_code"] in self.di.columns else COL["indication"]

    def _adr_id_col(self):
        return COL["meddra_code"] if COL["meddra_code"] in self.ar.columns else COL["adverse_event"]

    @staticmethod
    def _compute_edge_weight(count: int, mode: str) -> float:
        """
        Convert a raw co-occurrence count to an edge weight.

        mode:
          "log"  → log(count + 1)   — dampens high-frequency pairs
          "raw"  → count            — raw co-occurrence count
          "none" → 1.0              — uniform weight (topology only)
        """
        if mode == "log":
            return float(np.log(count + 1))
        elif mode == "raw":
            return float(count)
        else:  # "none"
            return 1.0

    def _build_ddi_edges(self, pids: list, threshold: int = 3) -> tuple:
        """
        Build Drug-Drug Interaction (DDI) edges from co-occurrence patterns.

        Edge weight mode is controlled by GRAPH["edge_weight_mode"] in config.py:
          "log"  → weight = log(count + 1)
          "raw"  → weight = count
          "none" → weight = 1.0 (topology only; note: HGTConv ignores edge_attr anyway)

        Returns:
            (edge_index, edge_attr) — edge_attr is None when mode is "none"
        """
        drug_col  = self._drug_id_col()
        pids_set  = set(pids)
        threshold = GRAPH.get("ddi_threshold", threshold)
        mode      = GRAPH.get("edge_weight_mode", "log")

        # Get all drugs per patient
        d_sub = self.drugs[self.drugs[COL["primaryid"]].isin(pids_set)].copy()
        d_sub["drug_id_str"] = d_sub[drug_col].astype(str)
        d_sub = d_sub[d_sub["drug_id_str"].isin(self.drug_vocab)]

        # Group by patient to get drug sets
        from collections import defaultdict
        patient_drugs = defaultdict(set)
        for _, row in d_sub.iterrows():
            patient_drugs[row[COL["primaryid"]]].add(row["drug_id_str"])

        # Count co-occurrences (unordered pairs)
        pair_counts = defaultdict(int)
        for drugs in patient_drugs.values():
            drugs_list = sorted(drugs)
            for i in range(len(drugs_list)):
                for j in range(i + 1, len(drugs_list)):
                    pair_counts[(drugs_list[i], drugs_list[j])] += 1

        # Build edge list
        edge_list    = []
        edge_weights = []
        for (d1, d2), count in pair_counts.items():
            if count >= threshold:
                idx1 = self.drug_vocab.get(d1)
                idx2 = self.drug_vocab.get(d2)
                if idx1 is not None and idx2 is not None:
                    edge_list.extend([[idx1, idx2], [idx2, idx1]])
                    w = self._compute_edge_weight(count, mode)
                    edge_weights.extend([w, w])

        if not edge_list:
            return (torch.zeros((2, 0), dtype=torch.long), None)

        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        edge_attr  = (None if mode == "none"
                      else torch.tensor(edge_weights, dtype=torch.float))
        return edge_index, edge_attr

    def _build_iii_edges(self, pids: list, threshold: int = 3) -> tuple:
        """
        Build Disease-Disease (Indication-Indication) co-occurrence edges.

        Edge weight mode is controlled by GRAPH["edge_weight_mode"] in config.py:
          "log"  → weight = log(count + 1)
          "raw"  → weight = count
          "none" → weight = 1.0 (topology only)

        Returns:
            (edge_index, edge_attr) — edge_attr is None when mode is "none"
        """
        dis_col   = self._disease_id_col()
        pids_set  = set(pids)
        threshold = GRAPH.get("iii_threshold", threshold)
        mode      = GRAPH.get("edge_weight_mode", "log")

        # Get all diseases per patient
        i_sub = self.di[self.di[COL["primaryid"]].isin(pids_set)].copy()
        i_sub["dis_id_str"] = i_sub[dis_col].astype(str)
        i_sub = i_sub[i_sub["dis_id_str"].isin(self.disease_vocab)]

        # Group by patient to get disease sets
        from collections import defaultdict
        patient_diseases = defaultdict(set)
        for _, row in i_sub.iterrows():
            patient_diseases[row[COL["primaryid"]]].add(row["dis_id_str"])

        # Count co-occurrences (unordered pairs)
        pair_counts = defaultdict(int)
        for diseases in patient_diseases.values():
            diseases_list = sorted(diseases)
            for i in range(len(diseases_list)):
                for j in range(i + 1, len(diseases_list)):
                    pair_counts[(diseases_list[i], diseases_list[j])] += 1

        # Build edge list
        edge_list    = []
        edge_weights = []
        for (dis1, dis2), count in pair_counts.items():
            if count >= threshold:
                idx1 = self.disease_vocab.get(dis1)
                idx2 = self.disease_vocab.get(dis2)
                if idx1 is not None and idx2 is not None:
                    edge_list.extend([[idx1, idx2], [idx2, idx1]])
                    w = self._compute_edge_weight(count, mode)
                    edge_weights.extend([w, w])

        if not edge_list:
            return (torch.zeros((2, 0), dtype=torch.long), None)

        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        edge_attr  = (None if mode == "none"
                      else torch.tensor(edge_weights, dtype=torch.float))
        return edge_index, edge_attr

    def _build_edges(self, pids: list, pid_to_node: dict,
                     include_adr_edges: bool,
                     include_ddi_edges: bool,
                     include_iii_edges: bool) -> dict:
        """
        Build edge index tensors for a subset of patients.
        Returns dict of edge_type → (edge_index, optional edge_attr)
        """
        pids_set = set(pids)
        drug_col = self._drug_id_col()
        dis_col  = self._disease_id_col()
        adr_col  = self._adr_id_col()

        # ── patient_takes_drug ─────────────────────────────────────────────
        d_sub = self.drugs[self.drugs[COL["primaryid"]].isin(pids_set)].copy()
        d_sub["drug_id_str"] = d_sub[drug_col].astype(str)
        d_sub = d_sub[d_sub["drug_id_str"].isin(self.drug_vocab)]
        p_idx_d = d_sub[COL["primaryid"]].map(pid_to_node).values
        d_idx   = d_sub["drug_id_str"].map(self.drug_vocab).values
        mask    = ~(np.isnan(p_idx_d.astype(float)) | np.isnan(d_idx.astype(float)))
        ptd_edge = torch.tensor(
            np.stack([p_idx_d[mask].astype(int), d_idx[mask].astype(int)]),
            dtype=torch.long)

        # ── patient_has_disease ─────────────────────────────────────────────
        i_sub = self.di[self.di[COL["primaryid"]].isin(pids_set)].copy()
        i_sub["dis_id_str"] = i_sub[dis_col].astype(str)
        i_sub = i_sub[i_sub["dis_id_str"].isin(self.disease_vocab)]
        p_idx_i = i_sub[COL["primaryid"]].map(pid_to_node).values
        i_idx   = i_sub["dis_id_str"].map(self.disease_vocab).values
        mask    = ~(np.isnan(p_idx_i.astype(float)) | np.isnan(i_idx.astype(float)))
        phi_edge = torch.tensor(
            np.stack([p_idx_i[mask].astype(int), i_idx[mask].astype(int)]),
            dtype=torch.long)

        edges = {
            ("patient", "takes", "drug")    : (ptd_edge,  None),
            ("drug",    "taken_by", "patient"): (ptd_edge[[1,0]], None),
            ("patient", "has",   "disease")  : (phi_edge,  None),
            ("disease", "had_by", "patient") : (phi_edge[[1,0]], None),
        }

        # ── drug-drug interaction edges (DDI) - TB datasets only ───────────
        if include_ddi_edges:
            ddi_edge_index, ddi_edge_attr = self._build_ddi_edges(pids)
            if ddi_edge_index.shape[1] > 0:
                edges[("drug", "interacts", "drug")] = (ddi_edge_index, ddi_edge_attr)
                print(f"    DDI edges: {ddi_edge_index.shape[1]:,} edges added")

        # ── disease-disease interaction edges (III) - TB datasets only ─────
        if include_iii_edges:
            iii_edge_index, iii_edge_attr = self._build_iii_edges(pids)
            if iii_edge_index.shape[1] > 0:
                edges[("disease", "comorbid", "disease")] = (iii_edge_index, iii_edge_attr)
                print(f"    III edges: {iii_edge_index.shape[1]:,} edges added")

        # ── Knowledge Graph edges (DrugBank DDI, drug-target, SIDER ADR) ─────
        if KG_EDGES.get("enabled", False) and self.kg_data:

            # Step 2: DrugBank pharmacological DDI
            if KG_EDGES.get("use_drugbank_ddi", False):
                kg_ddi_ei, _ = self._build_kg_ddi_edges()
                if kg_ddi_ei.shape[1] > 0:
                    edges[("drug", "db_interacts", "drug")] = (kg_ddi_ei, None)

            # Step 1: drug-target (only if target_vocab has been built)
            if KG_EDGES.get("use_drug_target", False) and self.target_vocab:
                dt_ei, _ = self._build_kg_drug_target_edges()
                if dt_ei.shape[1] > 0:
                    edges[("drug",   "has_target",  "target")] = (dt_ei,         None)
                    edges[("target", "targeted_by", "drug")]   = (dt_ei[[1, 0]], None)

            # Step 3: SIDER population-level drug-ADR prior
            if KG_EDGES.get("use_sider_adr", False):
                sa_ei, _ = self._build_kg_sider_adr_edges()
                if sa_ei.shape[1] > 0:
                    edges[("drug", "kg_has_adr", "adr")] = (sa_ei,         None)
                    edges[("adr",  "kg_adr_of",  "drug")] = (sa_ei[[1, 0]], None)

            # Step 4: KGIA disease comorbidity edges
            if KG_EDGES.get("use_kgia_comorbidity", False) and self.kgia_data:
                cm_ei, _ = self._build_kgia_comorbidity_edges()
                if cm_ei.shape[1] > 0:
                    edges[("disease", "kg_comorbid", "disease")] = (cm_ei, None)

            # Step 5: KGIA disease-risk factor edges
            if KG_EDGES.get("use_kgia_risk", False) and self.kgia_data:
                rk_ei, _ = self._build_kgia_risk_edges()
                if rk_ei.shape[1] > 0:
                    edges[("disease", "kg_has_risk", "disease")] = (rk_ei,         None)
                    edges[("disease", "kg_risk_of",  "disease")] = (rk_ei[[1, 0]], None)

        # ── Step 6: DrugCentral drug-target edges (independent of kg_data) ──
        if (KG_EDGES.get("enabled", False) and
                KG_EDGES.get("use_drugcentral_target", False) and self.dc_data):
            dc_ei, _ = self._build_drugcentral_target_edges()
            if dc_ei.shape[1] > 0:
                edges[("drug",   "dc_has_target",   "target")] = (dc_ei,         None)
                edges[("target", "dc_targeted_by",  "drug")]   = (dc_ei[[1, 0]], None)

        # ── Step 7: OnSIDES drug-label ADE edges (independent of kg_data) ─
        if (KG_EDGES.get("enabled", False) and
                KG_EDGES.get("use_onsides_adr", False) and self.onsides_data):
            os_ei, _ = self._build_onsides_adr_edges()
            if os_ei.shape[1] > 0:
                edges[("drug", "onsides_has_adr", "adr")] = (os_ei,         None)
                edges[("adr",  "onsides_adr_of",  "drug")] = (os_ei[[1, 0]], None)

        # ── drug-signals-adr edges from pharmacovigilance signals ─────────
        if SIGNAL_EDGES.get("enabled", False) and self.prop_edge_lookup:
            sig_ei, sig_ea = self._build_signal_edges()
            if sig_ei.shape[1] > 0:
                edges[("drug",  "signals_adr",  "adr")]  = (sig_ei,         sig_ea)
                edges[("adr",   "signaled_by",  "drug")] = (sig_ei[[1, 0]], sig_ea)
                print(f"    Signal edges: {sig_ei.shape[1]:,} drug→ADR + reverse added")

        # ── patient_experiences_adr (training labels only) ─────────────────
        if include_adr_edges:
            a_sub = self.ar[self.ar[COL["primaryid"]].isin(pids_set)].copy()
            a_sub["adr_id_str"] = a_sub[adr_col].astype(str)
            a_sub = a_sub[a_sub["adr_id_str"].isin(self.adr_vocab)]
            p_idx_a = a_sub[COL["primaryid"]].map(pid_to_node).values
            a_idx   = a_sub["adr_id_str"].map(self.adr_vocab).values
            mask    = ~(np.isnan(p_idx_a.astype(float)) | np.isnan(a_idx.astype(float)))
            pea_edge = torch.tensor(
                np.stack([p_idx_a[mask].astype(int), a_idx[mask].astype(int)]),
                dtype=torch.long)
            edges[("patient", "experiences", "adr")] = (pea_edge, None)
            edges[("adr", "experienced_by", "patient")] = (pea_edge[[1,0]], None)

        return edges

    # ── label matrix ─────────────────────────────────────────────────────────

    def _label_matrix(self, pids: list, pid_to_node: dict) -> torch.Tensor:
        """
        Multi-label binary matrix (N_patients × N_ADRs) for ADR prediction.
        pid_to_node maps patient IDs to their row indices in the label matrix.
        """
        adr_col = self._adr_id_col()
        n_adr   = len(self.adr_vocab)
        labels  = torch.zeros(len(pids), n_adr, dtype=torch.float)

        a_sub = self.ar[self.ar[COL["primaryid"]].isin(set(pids))].copy()
        a_sub["adr_id_str"] = a_sub[adr_col].astype(str)
        a_sub = a_sub[a_sub["adr_id_str"].isin(self.adr_vocab)]

        # Create a mapping from pid to label matrix row index (0-indexed within eval set)
        eval_pid_to_label_idx = {p: i for i, p in enumerate(pids)}
        
        for _, row in a_sub.iterrows():
            pid = row[COL["primaryid"]]
            label_idx = eval_pid_to_label_idx.get(pid)
            a_node = self.adr_vocab.get(row["adr_id_str"])
            if label_idx is not None and a_node is not None:
                labels[label_idx, a_node] = 1.0
        return labels

    # ── assemble HeteroData ───────────────────────────────────────────────────
    def _build_hetero_data(self, split_pids: dict,
                           all_train_pids: set,
                           include_split: str,
                           include_ddi: bool = False,
                           include_iii: bool = False) -> HeteroData:
        """
        Build a HeteroData graph for one split.
        split_pids  : {"train": set, "val": set, "test": set}
        include_split: "train" | "val" | "test"
            Only the split indicated gets ADR label edges included.
        """
        # All patients in the graph (train always present; val/test added at inference)
        if include_split == "train":
            active_pids = list(split_pids["train"])
        else:
            # At inference: train patients + target-split patients (no ADR edges for new)
            active_pids = list(split_pids["train"] | split_pids[include_split])

        pid_to_node = {p: i for i, p in enumerate(active_pids)}

        data = HeteroData()

        # Node features
        data["patient"].x  = self._patient_features(active_pids)
        data["drug"].x     = self._drug_features()
        data["disease"].x  = self._disease_features()
        data["adr"].x      = self._adr_features()
        if self.target_vocab:
            data["target"].x = self._target_features()

        # Store patient index (useful for label lookup at training time)
        data["patient"].pid = active_pids

        # Edges (ADR label edges for train only)
        train_pids_in_graph = split_pids["train"]
        edges = self._build_edges(
            pids=active_pids,
            pid_to_node=pid_to_node,
            include_adr_edges=(include_split == "train"),
            include_ddi_edges=include_ddi,
            include_iii_edges=include_iii
        )
        for edge_type, (ei, ea) in edges.items():
            src, rel, dst = edge_type
            data[src, rel, dst].edge_index = ei
            if ea is not None:
                data[src, rel, dst].edge_attr = ea

        # Label matrices (always built; used during training/evaluation)
        if include_split == "train":
            eval_pids = list(split_pids["train"])
        else:
            eval_pids = list(split_pids[include_split])

        # ── ADR-vocab coverage filter ──────────────────────────────────────
        # When ADR_FILTER is active (top_n_adrs or adr_list_file), some patients
        # in the split may have ZERO positive labels in the restricted vocab.
        # Including them in evaluation inflates the denominator of Hit@K and
        # deflates AUC (they can never contribute a true positive).
        # We only apply this filter when ADR_FILTER is actually restricting the
        # vocab; with the full vocab every patient has at least one label by
        # construction (QC-4 ensures each patient has ≥1 ADR).
        adr_filter_active = (
            ADR_FILTER.get("top_n_adrs") is not None
            or (ADR_FILTER.get("adr_list_file") and
                os.path.exists(str(ADR_FILTER.get("adr_list_file"))))
        )
        if adr_filter_active:
            adr_col   = self._adr_id_col()
            vocab_set = set(self.adr_vocab.keys())

            # Build set of patients who have ≥1 ADR inside the active vocab
            ar_sub = self.ar[self.ar[COL["primaryid"]].isin(set(eval_pids))].copy()
            ar_sub["_adr_str"] = ar_sub[adr_col].astype(str)
            covered_pids = set(
                ar_sub.loc[ar_sub["_adr_str"].isin(vocab_set), COL["primaryid"]]
            )

            before = len(eval_pids)
            eval_pids = [p for p in eval_pids if p in covered_pids]
            after  = len(eval_pids)
            if before != after:
                print(f"    [{include_split}] ADR-vocab filter: "
                      f"{before:,} → {after:,} eval patients "
                      f"({before - after:,} had no label in active vocab)")
        # ── end coverage filter ────────────────────────────────────────────

        # CRITICAL FIX: Build label matrix for ALL patients, not just eval patients
        # This ensures y.shape[0] == len(active_pids) == eval_mask.shape[0]
        data["patient"].y = self._label_matrix(active_pids, pid_to_node)

        # Mask to identify eval patients inside the full patient node list
        # (only covered patients are marked True when the filter is active)
        eval_mask = torch.zeros(len(active_pids), dtype=torch.bool)
        for p in eval_pids:
            if p in pid_to_node:
                eval_mask[pid_to_node[p]] = True
        data["patient"].eval_mask = eval_mask

        # Vocabulary metadata
        data.drug_vocab    = self.drug_vocab
        data.disease_vocab = self.disease_vocab
        data.adr_vocab     = self.adr_vocab
        if self.target_vocab:
            data.target_vocab = self.target_vocab

        return data

    # ── build & save one variant ──────────────────────────────────────────────
    def build_variant(self, variant: str):
        """
        variant: "xxx" | "xxx_gender" | "xxx_age"
        """
        print(f"\n{'='*70}")
        print(f"  BUILDING AER GRAPH: {variant.upper()}")
        print(f"{'='*70}")

        split_pids = self.splits[variant]
        train_pids = split_pids["train"]

        # Vocabulary built from training set only
        self._build_vocab(train_pids)
        self._build_prop_edge_lookup()      # populate self.prop_edge_lookup for signal edges
        self._build_kg_target_vocab()       # populate self.target_vocab from KG drug-target

        # DDI/III edges are now controlled explicitly via GRAPH config
        include_ddi = GRAPH.get("include_ddi", False)
        include_iii = GRAPH.get("include_iii", False)
        weight_mode = GRAPH.get("edge_weight_mode", "log")

        print(f"  Graph config: DDI={include_ddi}, III={include_iii}, "
              f"weight_mode='{weight_mode}'")

        out_dir = os.path.join(self.graph_path, variant)
        os.makedirs(out_dir, exist_ok=True)

        for split in ("train", "val", "test"):
            print(f"  Building {split} graph …")
            graph = self._build_hetero_data(split_pids, train_pids,
                                             include_split=split,
                                             include_ddi=include_ddi,
                                             include_iii=include_iii)
            out_path = os.path.join(out_dir, f"{split}_graph.pt")
            torch.save(graph, out_path)
            n_pat      = graph["patient"].x.shape[0]
            n_dr       = graph["drug"].x.shape[0]
            n_dis      = graph["disease"].x.shape[0]
            n_adr      = graph["adr"].x.shape[0]
            n_tgt      = graph["target"].x.shape[0] if "target" in graph.node_types else 0
            n_eval     = int(graph["patient"].eval_mask.sum())
            print(f"    ✓ Saved {out_path}")
            print(f"      Nodes: patient={n_pat:,} drug={n_dr:,} "
                  f"disease={n_dis:,} adr={n_adr:,} target={n_tgt:,} | "
                  f"eval_patients={n_eval:,}")
            print(f"      Edge types: {graph.edge_types}")

        # Save vocabularies as JSON for cross-step use
        vocab_path = os.path.join(out_dir, "vocabularies.json")
        vocab_data = {
            "drug_vocab"   : self.drug_vocab,
            "disease_vocab": self.disease_vocab,
            "adr_vocab"    : self.adr_vocab,
        }
        if self.target_vocab:
            vocab_data["target_vocab"] = self.target_vocab
        with open(vocab_path, "w") as f:
            json.dump(vocab_data, f, indent=2)
        print(f"  Vocabularies saved → {vocab_path}")

    # ── main entry ────────────────────────────────────────────────────────────

    def build_all(self, variants=("xxx", "xxx_gender", "xxx_age")):
        self.load()
        os.makedirs(self.graph_path, exist_ok=True)
        for v in variants:
            if v in self.splits:
                self.build_variant(v)
            else:
                print(f"  ⚠ Variant '{v}' not found in splits.json – skipping")
        print("\n✓ Step 2 (AER Graph Construction) complete!")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PreciseADR Step 2: AER Graph Construction")
    p.add_argument("--data-path",   default=DATA_PATH)
    p.add_argument("--preprocessed-path", default=PREPROCESSED_PATH)
    p.add_argument("--output-path", default=OUTPUT_PATH)
    p.add_argument("--graph-path",  default=GRAPH_PATH)
    p.add_argument("--variant",     default="all",
                   choices=["all", "xxx", "xxx_gender", "xxx_age"],
                   help="Which dataset variant to build (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    builder = AERGraphBuilder(args.data_path, args.preprocessed_path, args.output_path, args.graph_path)
    variants = (["xxx", "xxx_gender", "xxx_age"]
                if args.variant == "all" else [args.variant])
    builder.build_all(variants)


if __name__ == "__main__":
    main()
