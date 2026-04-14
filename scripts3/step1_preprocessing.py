"""
step1_preprocessing.py
======================
Step 1–2 of the PreciseADR reproduction pipeline - scripts3 version.

SCRIPTS3 UPDATE: Multi-dataset support for FAERS_ALL, FAERS_TB, FAERS_TB_DRUGS
  - Reads dataset from environment variable DATASET_NAME or config.py
  - Data paths automatically adjusted based on selected dataset
  - Supports all three dataset variants with same processing pipeline

Implements:
  1.0  Load all 11 Standardized FAERS tables
  1.1  (Optional) DrugBank ID mapping      → DRUGS_STANDARDIZED_DRUGBANK.csv
  1.2  (Optional) MedDRA ID mapping        → ADVERSE_REACTIONS_MEDDRA.csv
                                             DRUG_INDICATIONS_MEDDRA.csv
  2.1  Quality control (QC-1 … QC-4)
  2.2  Drug interference filtering         → XXX base dataset
  2.3  Dataset statistics tracking table
  2.4  Gender-related ADR association mining
  2.5  Age-related ADR association mining
  2.6  Build XXX / XXX-Gender / XXX-Age datasets
  2.7  Train / val / test split (stratified, seed=42)
  2.8  Save preprocessed datasets

Usage
-----
    # Using default dataset from config.py
    python step1_preprocessing.py

    # Selecting specific dataset
    DATASET_NAME=FAERS_TB python step1_preprocessing.py
    DATASET_NAME=FAERS_TB_DRUGS python step1_preprocessing.py

    # With optional mapping steps:
    python step1_preprocessing.py --drugbank-mapping ../data/drugname_drugbank_mapping.tsv
                                  --meddra-file      ../data/meddra.tsv
"""

import os
import sys
import json
import argparse
import warnings
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

warnings.filterwarnings("ignore")

# ── import shared config ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_PATH, PREPROCESSED_PATH, DRUGBANK_MAPPING_FILE_PATH, MEDDRA_MAPPING_FILE_PATH, 
    FAERS_SEP, TABLES, COL,
    QC, AGE_BINS, AGE_LABELS, ASSOC, SPLIT, OUTCOME_CODES,
    DATASET_NAME, print_config,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level helper for multiprocessing (must be picklable at module scope)
# ─────────────────────────────────────────────────────────────────────────────

def _gender_test(args):
    """Fisher's exact test for a single ADR's gender association."""
    adr, pid_gender_map, adr_pids = args
    try:
        f_with = f_without = m_with = m_without = 0
        for pid, g in pid_gender_map.items():
            has = pid in adr_pids
            if g == "F":
                f_with += has; f_without += not has
            elif g == "M":
                m_with += has; m_without += not has
        if f_with + f_without == 0 or m_with + m_without == 0:
            return None
        if f_with + m_with == 0 or f_without + m_without == 0:
            return None
        table = [[f_with, f_without], [m_with, m_without]]
        _, p_f = fisher_exact(table, alternative="greater")
        _, p_m = fisher_exact(table, alternative="less")
        or_val = (f_with * m_without) / (f_without * m_with) if f_without * m_with > 0 else float("inf")
        return {"ADVERSE_EVENT": adr, "p_female_higher": p_f, "p_male_higher": p_m,
                "odds_ratio": or_val,
                "f_with": f_with, "f_without": f_without,
                "m_with": m_with, "m_without": m_without}
    except Exception:
        return None


def _age_test(args):
    """Chi-square test for a single ADR's age-group association."""
    from scipy.stats import chi2_contingency
    adr, pid_age_map, adr_pids = args
    try:
        counts = {g: [0, 0] for g in AGE_LABELS}   # [with_adr, without_adr]
        for pid, grp in pid_age_map.items():
            has = pid in adr_pids
            counts[grp][0] += has; counts[grp][1] += not has
        table = [counts[g] for g in AGE_LABELS]
        # Need all groups to have at least some patients
        if any(sum(row) == 0 for row in table):
            return None
        if sum(row[0] for row in table) == 0:
            return None
        chi2, p, _, _ = chi2_contingency(table)
        # Risk ratio Elderly vs Youth
        y_tot = sum(counts["Youth"]);   y_risk = counts["Youth"][0] / y_tot  if y_tot else 0
        e_tot = sum(counts["Elderly"]); e_risk = counts["Elderly"][0] / e_tot if e_tot else 0
        rr = e_risk / y_risk if y_risk > 0 else float("inf")
        return {"ADVERSE_EVENT": adr, "p_value": p, "chi2": chi2, "risk_ratio": rr,
                "youth_with": counts["Youth"][0], "adult_with": counts["Adult"][0],
                "elderly_with": counts["Elderly"][0]}
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main processor
# ─────────────────────────────────────────────────────────────────────────────

class Step1Preprocessor:

    def __init__(self, data_path: str = DATA_PATH, output_path: str = PREPROCESSED_PATH):
        self.data_path   = data_path
        self.output_path = output_path
        self.ds: dict    = {}           # loaded DataFrames
        self.stats: dict = {}           # statistics at each stage
        self.pids: set   = set()        # active primaryid set

    # ── helpers ──────────────────────────────────────────────────────────────
    def _sep(self) -> str:
        return FAERS_SEP

    def _path(self, name: str) -> str:
        return os.path.join(self.data_path, TABLES[name])

    def _fallback_path(self, preferred: str, fallback: str) -> str:
        p = os.path.join(self.data_path, preferred)
        if os.path.exists(p):
            return p
        return os.path.join(self.data_path, fallback)

    def _record_stats(self, label: str):
        """Snapshot current dataset counts into self.stats."""
        pids = set(self.ds["demographics"][COL["primaryid"]].unique())
        self.stats[label] = {
            "case_reports"   : len(pids),
            "unique_drugs"   : self.ds["drugs_standardized"][COL["drug_name"]].nunique(),
            "unique_adrs"    : self.ds["adverse_reactions"][COL["adverse_event"]].nunique(),
            "unique_diseases": self.ds["drug_indications"][COL["indication"]].nunique(),
        }

    def _keep_pids(self, pids: set):
        """Filter every table to the given set of primaryids."""
        for name in ["demographics", "drugs_standardized", "adverse_reactions",
                     "drug_indications", "therapy_dates", "case_outcomes", "report_sources"]:
            if name in self.ds:
                self.ds[name] = self.ds[name][
                    self.ds[name][COL["primaryid"]].isin(pids)
                ].copy()
        self.pids = pids

    @staticmethod
    def _categorize_age(age: float) -> str:
        if age < 18:   return "Youth"
        if age < 65:   return "Adult"
        return "Elderly"

    # ── 1.0 Load data ────────────────────────────────────────────────────────

    def load_data(self):
        print("=" * 70)
        print("STEP 1.0  LOADING STANDARDIZED FAERS TABLES")
        print("=" * 70)
        sep = self._sep()

        # Core tables (always required)
        # core = {
        #     "demographics"      : TABLES["demographics"],
        #     "drugs_standardized": None,   # use fallback logic below
        #     "adverse_reactions" : None,
        #     "drug_indications"  : None,
        #     "therapy_dates"     : TABLES["therapy_dates"],
        #     "case_outcomes"     : TABLES["case_outcomes"],
        #     "report_sources"    : TABLES["report_sources"],
        # }
        core = {
            "demographics"      : TABLES["demographics"],
            "drugs_standardized": None,   # use fallback logic below
            "adverse_reactions" : TABLES["adverse_reactions"],
            "drug_indications"  : TABLES["drug_indications"],
            "therapy_dates"     : TABLES["therapy_dates"],
            "case_outcomes"     : TABLES["case_outcomes"],
            "report_sources"    : TABLES["report_sources"],
        }

        # Fallbacks for enriched tables
        # fallbacks = {
        #     "drugs_standardized": ("DRUGS_STANDARDIZED_DRUGBANK.csv", "DRUGS_STANDARDIZED.csv"),
        #     "adverse_reactions" : ("ADVERSE_REACTIONS_MEDDRA.csv", "ADVERSE_REACTIONS.csv"),
        #     "drug_indications"  : ("DRUG_INDICATIONS_MEDDRA.csv", "DRUG_INDICATIONS.csv"),
        # }
        fallbacks = {
            "drugs_standardized": ("DRUGS_STANDARDIZED_DRUGBANK.csv", "DRUGS_STANDARDIZED.csv"),
        }

        for name, filename in core.items():
            if filename is None:
                pref, fall = fallbacks[name]
                filepath = self._fallback_path(pref, fall)
            else:
                filepath = os.path.join(self.data_path, filename)
            try:
                print(f"  Loading {name} …")
                self.ds[name] = pd.read_csv(filepath, sep=sep, low_memory=False)
                self.ds[name].columns = self.ds[name].columns.str.upper()
                print(f"    ✓ {len(self.ds[name]):,} records from {os.path.basename(filepath)}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

        # Optional pre-computed tables (load if present; not critical)
        for name in ("proportionate_analysis", "contingency_table"):
            filepath = os.path.join(self.data_path, TABLES[name])
            if os.path.exists(filepath):
                self.ds[name] = pd.read_csv(filepath, sep=sep, low_memory=False)
                self.ds[name].columns = self.ds[name].columns.str.upper()
                print(f"  ✓ Loaded {name}: {len(self.ds[name]):,} records")

        self.pids = set(self.ds["demographics"][COL["primaryid"]].unique())
        self._record_stats("initial")
        print(f"\n  Total initial case reports: {len(self.pids):,}\n")

    # ── 1.1 Optional DrugBank mapping ────────────────────────────────────────
    def add_drugbank_mapping(self, mapping_file: str):
        """Map DRUG names to DrugBank_ID using a TSV file (DRUG_NAME, DRUGBANK_ID)."""
        print("=" * 70)
        print("STEP 1.1  DRUGBANK ID MAPPING (optional)")
        print("=" * 70)
        mapping = pd.read_csv(mapping_file, sep="\t")
        mapping.columns = mapping.columns.str.upper()
        mapping["_norm"] = mapping["DRUG_NAME"].str.lower().str.strip()
        self.ds["drugs_standardized"]["_norm"] = (
            self.ds["drugs_standardized"][COL["drug_name"]].str.lower().str.strip()
        )
        before = len(self.ds["drugs_standardized"])
        merged = self.ds["drugs_standardized"].merge(
            mapping[["_norm", "DRUGBANK_ID"]], on="_norm", how="left"
        ).drop(columns=["_norm"])
        mapped = merged["DRUGBANK_ID"].notna().sum()
        print(f"  Mapped: {mapped:,}/{before:,} records "
              f"({100*mapped/before:.1f}%)")
        self.ds["drugs_standardized"] = merged
        # Persist enriched file
        out = os.path.join(self.data_path, "DRUGS_STANDARDIZED_DRUGBANK.csv")
        merged.to_csv(out, sep=self._sep(), index=False)
        print(f"  Saved → {out}\n")

    # ── 1.2 Optional MedDRA mapping ──────────────────────────────────────────
    def add_meddra_mapping(self, meddra_file: str, term_type: str = "PT"):
        """
        Add MEDDRA_CODE to ADVERSE_REACTIONS and DRUG_INDICATIONS tables.

        meddra_file: TSV with no header, columns: UMLS_ID | term_type | MedDRA_ID | name
        """
        print("=" * 70)
        print("STEP 1.2  MEDDRA ID MAPPING (optional)")
        print("=" * 70)
        meddra = pd.read_csv(
            meddra_file, sep="\t", header=None,
            names=["UMLS_ID", "term_type", "MedDRA_ID", "name"],
            dtype={"MedDRA_ID": str},
        )
        if term_type:
            meddra = meddra[meddra["term_type"] == term_type].copy()
        meddra["_norm"] = meddra["name"].str.lower().str.strip()
        lookup = meddra.drop_duplicates("_norm")[["_norm", "MedDRA_ID"]]

        def _merge(df, term_col, label):
            df = df.copy()
            df["_norm"] = df[term_col].str.lower().str.strip()
            merged = df.merge(lookup, on="_norm", how="left").drop(columns=["_norm"])
            pct = 100 * merged["MedDRA_ID"].notna().sum() / len(df)
            print(f"  [{label}] mapped {pct:.1f}% of records")
            return merged

        self.ds["adverse_reactions"] = _merge(
            self.ds["adverse_reactions"], COL["adverse_event"], "ADVERSE_REACTIONS")
        self.ds["drug_indications"]  = _merge(
            self.ds["drug_indications"],  COL["indication"],    "DRUG_INDICATIONS")

        ar_out = os.path.join(self.data_path, "ADVERSE_REACTIONS_MEDDRA.csv")
        di_out = os.path.join(self.data_path, "DRUG_INDICATIONS_MEDDRA.csv")
        self.ds["adverse_reactions"].to_csv(ar_out, sep=self._sep(), index=False)
        self.ds["drug_indications"].to_csv( di_out, sep=self._sep(), index=False)
        print(f"  Saved → {ar_out}\n  Saved → {di_out}\n")

    # ── 2.1 Quality control ──────────────────────────────────────────────────
    def apply_quality_control(self):
        print("=" * 70)
        print("STEP 2.1  QUALITY CONTROL")
        print("=" * 70)

        # QC-1: US reports only
        us_pids = set(
            self.ds["demographics"].loc[
                self.ds["demographics"][COL["country_code"]] == QC["country_filter"],
                COL["primaryid"]
            ].unique()
        )
        print(f"  QC-1 (US only): {len(self.pids):,} → {len(us_pids):,}")
        self._keep_pids(us_pids)
        self._record_stats("after_qc1_us")

        # QC-2: Healthcare professional (HP) reports
        hp_pids = set(
            self.ds["report_sources"].loc[
                self.ds["report_sources"][COL["rpsr_cod"]] == QC["reporter_filter"],
                COL["primaryid"]
            ].unique()
        )
        us_hp_pids = us_pids & hp_pids
        print(f"  QC-2 (HP only): {len(us_pids):,} → {len(us_hp_pids):,}")
        self._keep_pids(us_hp_pids)
        self._record_stats("after_qc2_hp")

        # QC-3: Frequency filtering (> min_frequency occurrences)
        min_freq = QC["min_frequency"]
        freq_drugs = set(self.ds["drugs_standardized"][COL["drug_name"]].value_counts().loc[lambda x: x > min_freq].index)
        freq_adrs = set(self.ds["adverse_reactions"][COL["adverse_event"]].value_counts().loc[lambda x: x > min_freq].index)
        freq_diseases = set(self.ds["drug_indications"][COL["indication"]].value_counts().loc[lambda x: x > min_freq].index)
        print(f"  QC-3 frequency >{min_freq}: "
              f"drugs {self.ds['drugs_standardized'][COL['drug_name']].nunique():,}→{len(freq_drugs):,} | "
              f"ADRs {self.ds['adverse_reactions'][COL['adverse_event']].nunique():,}→{len(freq_adrs):,} | "
              f"diseases {self.ds['drug_indications'][COL['indication']].nunique():,}→{len(freq_diseases):,}")
        self.ds["drugs_standardized"] = self.ds["drugs_standardized"][
            self.ds["drugs_standardized"][COL["drug_name"]].isin(freq_drugs)]
        self.ds["adverse_reactions"]  = self.ds["adverse_reactions"][
            self.ds["adverse_reactions"][COL["adverse_event"]].isin(freq_adrs)]
        self.ds["drug_indications"]   = self.ds["drug_indications"][
            self.ds["drug_indications"][COL["indication"]].isin(freq_diseases)]
        self._record_stats("after_qc3_freq")

        # QC-4: Remove incomplete records (must have drugs AND ADRs AND diseases)
        d_pids = set(self.ds["drugs_standardized"][COL["primaryid"]].unique())
        a_pids = set(self.ds["adverse_reactions"][COL["primaryid"]].unique())
        i_pids = set(self.ds["drug_indications"][COL["primaryid"]].unique())
        complete_pids = d_pids & a_pids & i_pids
        print(f"  QC-4 (complete records): {len(self.pids):,} → {len(complete_pids):,}")
        self._keep_pids(complete_pids)
        self._record_stats("after_qc4_complete")

        print(f"  After QC: {len(self.pids):,} case reports\n")

    # ── 2.2 Drug interference filtering ──────────────────────────────────────
    def apply_drug_interference_filtering(self):
        print("=" * 70)
        print("STEP 2.2  DRUG INTERFERENCE FILTERING")
        print("=" * 70)

        exclusion_patterns = [
            # Administration errors
            "wrong drug", "incorrect dose", "medication error", "overdose",
            "underdose", "drug administered to patient of inappropriate age",
            "drug administered at inappropriate site", "expired drug administered",
            "drug administration error", "drug dispensing error",
            "accidental exposure", "wrong patient", "dose omission",
            "extra dose administered",
            # Product quality / counterfeit
            "product counterfeit", "counterfeit product",
            "counterfeit drug administered", "product packaging counterfeit",
            "product measured potency issue", "product label issue",
            "poor quality drug administered", "product contamination",
            "product quality issue", "product physical issue",
            "product shape issue", "product substitution issue",
            "incorrect product storage", "wrong product stored",
            "product colour issue", "product odour abnormal",
            "product leakage", "product name confusion",
            # Device-related
            "device failure", "device breakage", "device issue",
            "device misuse", "device malfunction", "device dislocation",
            "medical device site reaction", "pacemaker malfunction",
            "pacemaker complication", "cardiac pacemaker insertion",
            "stent occlusion", "stent placement", "pump reservoir issue",
            "intrathecal pump insertion", "catheter related complication",
            "injury associated with device",
            # Procedural / surgical
            "procedural complication", "surgical procedure repeated",
            "post procedural complication", "implantation complication",
            "anaesthetic complication",
            # Intentional harm
            "intentional overdose", "intentional product misuse",
            "intentional drug misuse", "suicide attempt",
            "self-injurious ideation", "intentional self-injury",
            "poisoning deliberate", "substance abuse",
            "drug abuse", "drug diversion",
            # Lack of information
            "no adverse event", "no reaction on previous exposure to drug",
            "unevaluable event", "product use issue",
            "circumstance or information capable of leading to medication error",
            # Non-specific / vague
            "product complaint", "off label use",
            "patient dissatisfaction", "concern",
        ]
        pats_lower = [p.lower() for p in exclusion_patterns]

        def _excluded(adr_text):
            if pd.isna(adr_text):
                return False
            t = str(adr_text).lower()
            return any(p in t for p in pats_lower)

        mask = self.ds["adverse_reactions"][COL["adverse_event"]].apply(_excluded)
        excluded_n = mask.sum()
        self.ds["adverse_reactions"] = self.ds["adverse_reactions"][~mask].copy()

        valid_pids = set(self.ds["adverse_reactions"][COL["primaryid"]].unique())
        self._keep_pids(valid_pids)
        self._record_stats("after_drug_interference")

        print(f"  Excluded {excluded_n:,} ADR records matching interference patterns")
        print(f"  Base dataset (XXX): {len(self.pids):,} case reports\n")

    # ── 2.3 Statistics tracking table ────────────────────────────────────────
    def print_statistics_table(self):
        print("=" * 70)
        print("STEP 2.3  DATASET STATISTICS TRACKING TABLE")
        print("=" * 70)
        stages = [
            ("Initial",                  "initial"),
            ("After US filter (QC-1)",   "after_qc1_us"),
            ("After HP filter (QC-2)",   "after_qc2_hp"),
            ("After freq filter (QC-3)", "after_qc3_freq"),
            ("After incomplete (QC-4)",  "after_qc4_complete"),
            ("After drug interference",  "after_drug_interference"),
        ]
        hdr = f"{'Filtering Step':<32} {'#Reports':>10} {'#Drugs':>10} {'#ADRs':>8} {'#Diseases':>10}"
        print(hdr)
        print("-" * len(hdr))
        for label, key in stages:
            if key in self.stats:
                s = self.stats[key]
                print(f"{label:<32} {s['case_reports']:>10,} {s['unique_drugs']:>10,}"
                      f" {s['unique_adrs']:>8,} {s['unique_diseases']:>10,}")
        print()

    # ── 2.4 Gender-related ADR mining ────────────────────────────────────────
    def identify_gender_related_adrs(self):
        print("=" * 70)
        print("STEP 2.4  GENDER-RELATED ADR ASSOCIATION MINING")
        print("=" * 70)
        alpha = ASSOC["alpha"]
        min_co = ASSOC["min_cooccurrence"]

        # Build patient→gender map (valid genders only)
        demo = self.ds["demographics"].copy()
        demo = demo[demo[COL["gender"]].isin(["M", "F"])]
        pid_gender = dict(zip(demo[COL["primaryid"]], demo[COL["gender"]]))
        valid_pids = set(pid_gender.keys())

        ar = self.ds["adverse_reactions"][
            self.ds["adverse_reactions"][COL["primaryid"]].isin(valid_pids)
        ].copy()

        # Patient-level contingency: for each ADR, count F_with/F_without/M_with/M_without
        adrs = ar[COL["adverse_event"]].unique()

        # Pre-compute ADR → set(pids)
        adr_pids_map = {
            adr: set(ar.loc[ar[COL["adverse_event"]] == adr, COL["primaryid"]])
            for adr in adrs
        }

        # Pre-filter: only test ADRs with >= min_co reports in any gender
        filtered_adrs = [
            adr for adr, pids in adr_pids_map.items()
            if len(pids) >= min_co
        ]
        print(f"  Testing {len(filtered_adrs):,} ADRs (≥{min_co} reports) …")

        args = [(adr, pid_gender, adr_pids_map[adr]) for adr in filtered_adrs]
        n_cores = max(1, cpu_count() - 1)
        with Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(_gender_test, args, chunksize=50),
                                total=len(args), desc="  Gender Fisher"))
        results = [r for r in results if r is not None]
        res_df = pd.DataFrame(results)

        # FDR correction
        res_df["p_min"] = res_df[["p_female_higher", "p_male_higher"]].min(axis=1)
        reject, p_corr, _, _ = multipletests(
            res_df["p_min"], alpha=alpha, method=ASSOC["correction"])
        res_df["p_corrected"] = p_corr
        res_df["is_gender_related"] = reject & (res_df["odds_ratio"] != 1.0)

        gender_adrs = res_df.loc[res_df["is_gender_related"], "ADVERSE_EVENT"].unique()
        print(f"  Gender-related ADRs found: {len(gender_adrs):,} / {len(filtered_adrs):,}")

        self.ds["gender_related_adrs"]  = gender_adrs
        self.ds["gender_test_results"]  = res_df
        return gender_adrs

    # ── 2.5 Age-related ADR mining ───────────────────────────────────────────
    def identify_age_related_adrs(self):
        print("=" * 70)
        print("STEP 2.5  AGE-RELATED ADR ASSOCIATION MINING")
        print("=" * 70)
        alpha = ASSOC["alpha"]
        min_co = ASSOC["min_cooccurrence"]

        demo = self.ds["demographics"].copy()
        demo = demo[
            (demo[COL["age"]] >= 0) & (demo[COL["age"]] <= 120)
        ].copy()
        demo["age_group"] = pd.cut(
            demo[COL["age"]],
            bins=AGE_BINS, labels=AGE_LABELS, right=False
        ).astype(str)
        pid_age = dict(zip(demo[COL["primaryid"]], demo["age_group"]))
        valid_pids = set(pid_age.keys())

        ar = self.ds["adverse_reactions"][
            self.ds["adverse_reactions"][COL["primaryid"]].isin(valid_pids)
        ].copy()

        adrs = ar[COL["adverse_event"]].unique()
        adr_pids_map = {
            adr: set(ar.loc[ar[COL["adverse_event"]] == adr, COL["primaryid"]])
            for adr in adrs
        }
        filtered_adrs = [adr for adr, pids in adr_pids_map.items()
                         if len(pids) >= min_co]
        print(f"  Testing {len(filtered_adrs):,} ADRs (≥{min_co} reports) …")

        args = [(adr, pid_age, adr_pids_map[adr]) for adr in filtered_adrs]
        n_cores = max(1, cpu_count() - 1)
        with Pool(n_cores) as pool:
            results = list(tqdm(pool.imap(_age_test, args, chunksize=50),
                                total=len(args), desc="  Age Chi-sq"))
        results = [r for r in results if r is not None]
        res_df = pd.DataFrame(results)

        reject, p_corr, _, _ = multipletests(
            res_df["p_value"], alpha=alpha, method=ASSOC["correction"])
        res_df["p_corrected"] = p_corr
        res_df["is_age_related"] = reject

        age_adrs = res_df.loc[res_df["is_age_related"], "ADVERSE_EVENT"].unique()
        print(f"  Age-related ADRs found: {len(age_adrs):,} / {len(filtered_adrs):,}")

        self.ds["age_related_adrs"]  = age_adrs
        self.ds["age_test_results"]  = res_df
        return age_adrs

    # ── 2.6 Build specialized datasets ───────────────────────────────────────
    def build_specialized_datasets(self):
        """Build XXX, XXX-Gender, and XXX-Age patient sets and 
        Compute statistics (case reports, unique drugs, ADRs, diseases) for each variant.
        """
        print("=" * 70)
        print("STEP 2.6  BUILD SPECIALIZED DATASETS")
        print("=" * 70)

        all_pids = list(self.pids)

        gender_adrs = set(self.ds.get("gender_related_adrs", []))
        age_adrs    = set(self.ds.get("age_related_adrs",    []))

        # XXX-Gender: patients who have at least one gender-related ADR
        if gender_adrs:
            gender_pids = set(
                self.ds["adverse_reactions"].loc[
                    self.ds["adverse_reactions"][COL["adverse_event"]].isin(gender_adrs),
                    COL["primaryid"]
                ].unique()
            ) & self.pids
        else:
            gender_pids = set()

        # XXX-Age: patients who have at least one age-related ADR
        if age_adrs:
            age_pids = set(
                self.ds["adverse_reactions"].loc[
                    self.ds["adverse_reactions"][COL["adverse_event"]].isin(age_adrs),
                    COL["primaryid"]
                ].unique()
            ) & self.pids
        else:
            age_pids = set()

        self.ds["xxx_pids"]        = set(all_pids)
        self.ds["xxx_gender_pids"] = gender_pids
        self.ds["xxx_age_pids"]    = age_pids

        # ── Calculate statistics for each variant ──────────────────────────────
        def _variant_stats(pids: set, adr_filter: set = None) -> dict:
            """
            Count unique drugs, ADRs, and diseases for a given patient set.

            pids       : the patient IDs in this variant
            adr_filter : if provided, count only these ADRs (not all ADRs of
                        those patients). Used for XXX-Gender and XXX-Age where
                        the defining feature is the ADR subset, not all ADRs.
            """
            drugs_sub = self.ds["drugs_standardized"][
                self.ds["drugs_standardized"][COL["primaryid"]].isin(pids)
            ]
            ar_sub = self.ds["adverse_reactions"][
                self.ds["adverse_reactions"][COL["primaryid"]].isin(pids)
            ]
            if adr_filter is not None:
                # Count only the defining ADRs, not every ADR of those patients
                ar_sub = ar_sub[ar_sub[COL["adverse_event"]].isin(adr_filter)]

            di_sub = self.ds["drug_indications"][
                self.ds["drug_indications"][COL["primaryid"]].isin(pids)
            ]
            return {
                "case_reports"   : len(pids),
                "unique_drugs"   : int(drugs_sub[COL["drug_name"]].nunique()),
                "unique_adrs"    : int(ar_sub[COL["adverse_event"]].nunique()),
                "unique_diseases": int(di_sub[COL["indication"]].nunique()),
            }
        
        stats_xxx        = _variant_stats(set(all_pids))
        stats_xxx_gender = _variant_stats(gender_pids, adr_filter=gender_adrs)
        stats_xxx_age    = _variant_stats(age_pids, adr_filter=age_adrs)

        # Store for later (used by save() → preprocessing_stats.json)
        self.stats["xxx"]        = stats_xxx
        self.stats["xxx_gender"] = stats_xxx_gender
        self.stats["xxx_age"]    = stats_xxx_age

        # ── Print statistics table ─────────────────────────────────────────────
        hdr = (f"  {'Dataset':<15} {'#Reports':>10} {'#Drugs':>8} "
            f"{'#ADRs':>8} {'#Diseases':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for label, s in [("XXX (full)",  stats_xxx),
                        ("XXX-Gender",  stats_xxx_gender),
                        ("XXX-Age",     stats_xxx_age)]:
            print(f"  {label:<15} {s['case_reports']:>10,} {s['unique_drugs']:>8,} "
                f"{s['unique_adrs']:>8,} {s['unique_diseases']:>10,}")
        print()

    # ── 2.7 Train / val / test split ─────────────────────────────────────────
    def split_datasets(self):
        """
        Stratified split (75/12.5/12.5) per dataset variant.
        Stratification is based on the most frequent ADR per patient.
        """
        print("=" * 70)
        print("STEP 2.7  TRAIN / VAL / TEST SPLIT (75/12.5/12.5, seed=42)")
        print("=" * 70)
        seed = SPLIT["seed"]

        # Build patient→most-frequent-ADR map for stratification
        adr_per_pid = (
            self.ds["adverse_reactions"]
            .groupby(COL["primaryid"])[COL["adverse_event"]]
            .agg(lambda x: x.value_counts().index[0])
        )

        splits = {}
        for name, pids in [
            ("xxx",        self.ds["xxx_pids"]),
            ("xxx_gender", self.ds["xxx_gender_pids"]),
            ("xxx_age",    self.ds["xxx_age_pids"]),
        ]:
            pids_list = sorted(pids)
            labels    = adr_per_pid.reindex(pids_list).fillna("__unknown__").tolist()

            # Two-step split: (train) vs (val+test)
            try:
                train_ids, valtest_ids, _, valtest_labels = train_test_split(
                    pids_list, labels,
                    test_size=SPLIT["val"] + SPLIT["test"],
                    stratify=labels, random_state=seed,
                )
                val_ids, test_ids = train_test_split(
                    valtest_ids,
                    test_size=SPLIT["test"] / (SPLIT["val"] + SPLIT["test"]),
                    stratify=valtest_labels, random_state=seed,
                )
            except ValueError:
                # If stratification fails (too few samples per class), do random
                train_ids, valtest_ids = train_test_split(
                    pids_list, test_size=SPLIT["val"] + SPLIT["test"],
                    random_state=seed)
                val_ids, test_ids = train_test_split(
                    valtest_ids,
                    test_size=SPLIT["test"] / (SPLIT["val"] + SPLIT["test"]),
                    random_state=seed)

            splits[name] = {
                "train": set(train_ids),
                "val"  : set(val_ids),
                "test" : set(test_ids),
            }
            print(f"  {name:12s}: total={len(pids_list):,} "
                  f"train={len(train_ids):,} "
                  f"val={len(val_ids):,} "
                  f"test={len(test_ids):,}")

        self.ds["splits"] = splits
        return splits

    # ── 2.8 Save preprocessed data ───────────────────────────────────────────
    def save(self):
        print("=" * 70)
        print("STEP 2.8  SAVING PREPROCESSED DATA")
        print("=" * 70)
        os.makedirs(self.output_path, exist_ok=True)
        sep = self._sep()

        # Save filtered core tables
        for name in ["demographics", "drugs_standardized", "adverse_reactions",
                     "drug_indications", "therapy_dates", "case_outcomes", "report_sources"]:
            if name in self.ds and isinstance(self.ds[name], pd.DataFrame):
                out = os.path.join(self.output_path, f"{name.upper()}_FILTERED.csv")
                self.ds[name].to_csv(out, sep=sep, index=False)
                print(f"  Saved {name} → {out}")

        # Save splits
        if "splits" in self.ds:
            splits_serializable = {
                variant: {split: [int(p) for p in pids]
                        for split, pids in variant_splits.items()}
                for variant, variant_splits in self.ds["splits"].items()
            }
            splits_path = os.path.join(self.output_path, "splits.json")
            with open(splits_path, "w") as f:
                json.dump(splits_serializable, f, indent=2)
            print(f"  Saved splits → {splits_path}")

        # Save gender / age related ADRs
        for key in ("gender_related_adrs", "age_related_adrs"):
            if key in self.ds:
                out = os.path.join(self.output_path, f"{key}.json")
                with open(out, "w") as f:
                    json.dump(list(self.ds[key]), f, indent=2)
                print(f"  Saved {key} → {out}")

        # Save statistics
        stats_path = os.path.join(self.output_path, "preprocessing_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"  Saved statistics → {stats_path}\n")
        
        for table_name in ("PROPORTIONATE_ANALYSIS.csv", "CONTINGENCY_TABLE.csv"):
            src = os.path.join(self.data_path, table_name)
            dst = os.path.join(self.output_path, table_name)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  Copied {table_name} → {dst}") 
            else:
                print(f"  {table_name} not found in {self.data_path} - skipping copy")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PreciseADR Step 1: Preprocessing")
    p.add_argument("--data-path", default=DATA_PATH)
    p.add_argument("--output-path", default=PREPROCESSED_PATH)
    p.add_argument("--drugbank-mapping", default=DRUGBANK_MAPPING_FILE_PATH,
                   help="Path to drugname_drugbank_mapping.tsv (optional)")
    p.add_argument("--meddra-file", default=MEDDRA_MAPPING_FILE_PATH,
                   help="Path to meddra.tsv (optional)")
    p.add_argument("--skip-gender-mining", action="store_true")
    p.add_argument("--skip-age-mining", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Print configuration
    print_config()
    print()

    proc = Step1Preprocessor(args.data_path, args.output_path)

    # 1.0 Load
    proc.load_data()

    # 1.1 Optional enrichment
    if args.drugbank_mapping:
        proc.add_drugbank_mapping(args.drugbank_mapping)
    # if args.meddra_file:
    #     proc.add_meddra_mapping(args.meddra_file)

    # 2.1–2.2 Filtering
    proc.apply_quality_control()
    proc.apply_drug_interference_filtering()
    proc.print_statistics_table()

    # 2.4–2.5 Association mining
    if not args.skip_gender_mining:
        proc.identify_gender_related_adrs()
    if not args.skip_age_mining:
        proc.identify_age_related_adrs()

    # 2.6–2.8 Finalize
    proc.build_specialized_datasets()
    proc.split_datasets()
    proc.save()

    print("✓ Step 1 complete!")
    return proc


if __name__ == "__main__":
    main()
