"""
build_datasets.py
=================
Build FAERS_ALL, FAERS_TB, and FAERS_TB_DRUGS datasets from standardized FAERS data.

This script creates three dataset variants:
  1. FAERS_ALL: Full dataset from standardized FAERS 2018-2025
  2. FAERS_TB: Tuberculosis-related cases (using FAERS_TB_PRIMARYID_SET)
  3. FAERS_TB_DRUGS: TB cases with TB-specific drugs (using FAERS_TB_DRUGS_PRIMARYID_SET)

Each dataset is stored in a separate folder:
  - data/FAERS_ALL/
  - data/FAERS_TB/
  - data/FAERS_TB_DRUGS/

Usage
-----
    python build_datasets.py
    python build_datasets.py --dataset all         # build all three datasets
    python build_datasets.py --dataset FAERS_TB    # build only FAERS_TB
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_PATH = os.path.join(PROJECT_ROOT, "../data/standardized_faers_2015_2025")
OUTPUT_DATA_PATH = os.path.join(PROJECT_ROOT, "../data")

# Separator
FAERS_SEP = ","

class DatasetBuilder:
    """Build FAERS dataset variants (ALL, TB, TB_DRUGS)."""
    
    def __init__(self, source_path: str, output_path: str):
        self.source_path = source_path
        self.output_path = output_path
        
    def build_faers_all(self):
        """Build FAERS_ALL dataset - full standardized FAERS data."""
        print("=" * 70)
        print("BUILDING FAERS_ALL DATASET")
        print("=" * 70)
        
        output_dir = os.path.join(self.output_path, "FAERS_ALL")
        os.makedirs(output_dir, exist_ok=True)
        
        # Core tables to copy
        tables = [
            "DEMOGRAPHICS.csv",
            "DRUGS_STANDARDIZED.csv",
            "ADVERSE_REACTIONS.csv",
            "DRUG_INDICATIONS.csv",
            "THERAPY_DATES.csv",
            "CASE_OUTCOMES.csv",
            "REPORT_SOURCES.csv",
            "PROPORTIONATE_ANALYSIS.csv",
            "CONTINGENCY_TABLE.csv",
            "DRUG_ADVERSE_REACTIONS_COUNT.csv",
        ]
        
        stats = {}
        
        for table_name in tables:
            source_file = os.path.join(self.source_path, table_name)
            if not os.path.exists(source_file):
                print(f"  [SKIP] {table_name} not found")
                continue
                
            print(f"  Processing {table_name} ...")
            df = pd.read_csv(source_file, sep=FAERS_SEP, low_memory=False)
            df.columns = df.columns.str.upper()
            
            # Save to output
            output_file = os.path.join(output_dir, table_name)
            df.to_csv(output_file, sep=FAERS_SEP, index=False)
            print(f"    ✓ Saved {len(df):,} records → {table_name}")
            
            # Track statistics
            if table_name == "DEMOGRAPHICS.csv":
                stats['case_reports'] = len(df)
            elif table_name == "DRUGS_STANDARDIZED.csv":
                stats['unique_drugs'] = df['DRUG'].nunique()
            elif table_name == "ADVERSE_REACTIONS.csv":
                stats['unique_adrs'] = df['ADVERSE_EVENT'].nunique()
            elif table_name == "DRUG_INDICATIONS.csv":
                stats['unique_diseases'] = df['DRUG_INDICATION'].nunique()
        
        print(f"\n  FAERS_ALL Statistics:")
        print(f"    Case Reports: {stats.get('case_reports', 'N/A')}")
        print(f"    Unique Drugs: {stats.get('unique_drugs', 'N/A')}")
        print(f"    Unique ADRs:  {stats.get('unique_adrs', 'N/A')}")
        print(f"    Unique Diseases: {stats.get('unique_diseases', 'N/A')}")
        print(f"\n  ✓ FAERS_ALL dataset saved to: {output_dir}\n")
        
        return stats
    
    def build_faers_tb(self):
        """Build FAERS_TB dataset - TB-related cases only."""
        print("=" * 70)
        print("BUILDING FAERS_TB DATASET")
        print("=" * 70)
        
        output_dir = os.path.join(self.output_path, "FAERS_TB")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load TB primaryid set
        tb_primaryid_file = os.path.join(self.source_path, "FAERS_TB_BASE_POPULATION.csv")
        if not os.path.exists(tb_primaryid_file):
            print(f"  ✗ ERROR: {tb_primaryid_file} not found!")
            return None
            
        tb_primaryids = pd.read_csv(tb_primaryid_file)['primaryid'].unique()
        print(f"  Loaded {len(tb_primaryids):,} TB-related case reports")
        
        # Tables to filter by primaryid
        tables = [
            "DEMOGRAPHICS.csv",
            "DRUGS_STANDARDIZED.csv",
            "ADVERSE_REACTIONS.csv",
            "DRUG_INDICATIONS.csv",
            "THERAPY_DATES.csv",
            "CASE_OUTCOMES.csv",
            "REPORT_SOURCES.csv",
        ]
        
        stats = {}
        
        for table_name in tables:
            source_file = os.path.join(self.source_path, table_name)
            if not os.path.exists(source_file):
                print(f"  [SKIP] {table_name} not found")
                continue
                
            print(f"  Processing {table_name} ...")
            df = pd.read_csv(source_file, sep=FAERS_SEP, low_memory=False)
            df.columns = df.columns.str.upper()
            
            # Filter by TB primaryids
            df_filtered = df[df['PRIMARYID'].isin(tb_primaryids)].copy()
            
            # Save to output
            output_file = os.path.join(output_dir, table_name)
            df_filtered.to_csv(output_file, sep=FAERS_SEP, index=False)
            print(f"    ✓ Saved {len(df_filtered):,} records → {table_name}")
            
            # Track statistics
            if table_name == "DEMOGRAPHICS.csv":
                stats['case_reports'] = len(df_filtered)
            elif table_name == "DRUGS_STANDARDIZED.csv":
                if 'unique_drugs' not in stats:
                    stats['unique_drugs'] = df_filtered['DRUG'].nunique()
            elif table_name == "ADVERSE_REACTIONS.csv":
                stats['unique_adrs'] = df_filtered['ADVERSE_EVENT'].nunique()
            elif table_name == "DRUG_INDICATIONS.csv":
                stats['unique_diseases'] = df_filtered['DRUG_INDICATION'].nunique()
        
        # Copy TB-specific analysis tables
        # tb_tables = [
        #     "FAERS_TB_PROPORTIONATE_ANALYSIS.csv",
        #     "FAERS_TB_CONTINGENCY_TABLE.csv",
        #     "FAERS_TB_DRUG_ADVERSE_REACTIONS_COUNT.csv",
        # ]
        tb_tables = [
            "PROPORTIONATE_ANALYSIS.csv",
            "CONTINGENCY_TABLE.csv",
            "DRUG_ADVERSE_REACTIONS_COUNT.csv",
        ]
        
        for table_name in tb_tables:
            source_file = os.path.join(self.source_path, table_name)
            if os.path.exists(source_file):
                df = pd.read_csv(source_file, sep=FAERS_SEP, low_memory=False)
                df.columns = df.columns.str.upper()
                # Rename to standard names
                new_name = table_name.replace("FAERS_TB_", "")
                output_file = os.path.join(output_dir, new_name)
                df.to_csv(output_file, sep=FAERS_SEP, index=False)
                print(f"    ✓ Copied {len(df):,} records → {new_name}")
        
        print(f"\n  FAERS_TB Statistics:")
        print(f"    Case Reports: {stats.get('case_reports', 'N/A'):,}")
        print(f"    Unique Drugs: {stats.get('unique_drugs', 'N/A'):,}")
        print(f"    Unique ADRs:  {stats.get('unique_adrs', 'N/A'):,}")
        print(f"    Unique Diseases: {stats.get('unique_diseases', 'N/A'):,}")
        print(f"\n  ✓ FAERS_TB dataset saved to: {output_dir}\n")
        
        return stats
    
    def build_faers_tb_drugs(self):
        """Build FAERS_TB_DRUGS dataset - TB cases with TB-specific drugs only."""
        print("=" * 70)
        print("BUILDING FAERS_TB_DRUGS DATASET")
        print("=" * 70)
        
        output_dir = os.path.join(self.output_path, "FAERS_TB_DRUGS")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load TB_DRUGS primaryid set
        tb_drugs_primaryid_file = os.path.join(self.source_path, "FAERS_TB_DRUGS_BASE_POPULATION.csv")
        if not os.path.exists(tb_drugs_primaryid_file):
            print(f"  ✗ ERROR: {tb_drugs_primaryid_file} not found!")
            return None
            
        tb_drugs_primaryids = pd.read_csv(tb_drugs_primaryid_file)['primaryid'].unique()
        print(f"  Loaded {len(tb_drugs_primaryids):,} TB-drugs-related case reports")
        
        # Tables to filter by primaryid
        tables = [
            "DEMOGRAPHICS.csv",
            "DRUGS_STANDARDIZED.csv",
            "ADVERSE_REACTIONS.csv",
            "DRUG_INDICATIONS.csv",
            "THERAPY_DATES.csv",
            "CASE_OUTCOMES.csv",
            "REPORT_SOURCES.csv",
        ]
        
        stats = {}
        
        for table_name in tables:
            source_file = os.path.join(self.source_path, table_name)
            if not os.path.exists(source_file):
                print(f"  [SKIP] {table_name} not found")
                continue
                
            print(f"  Processing {table_name} ...")
            df = pd.read_csv(source_file, sep=FAERS_SEP, low_memory=False)
            df.columns = df.columns.str.upper()
            
            # Filter by TB_DRUGS primaryids
            df_filtered = df[df['PRIMARYID'].isin(tb_drugs_primaryids)].copy()
            
            # Save to output
            output_file = os.path.join(output_dir, table_name)
            df_filtered.to_csv(output_file, sep=FAERS_SEP, index=False)
            print(f"    ✓ Saved {len(df_filtered):,} records → {table_name}")
            
            # Track statistics
            if table_name == "DEMOGRAPHICS.csv":
                stats['case_reports'] = len(df_filtered)
            elif table_name == "DRUGS_STANDARDIZED.csv":
                if 'unique_drugs' not in stats:
                    stats['unique_drugs'] = df_filtered['DRUG'].nunique()
            elif table_name == "ADVERSE_REACTIONS.csv":
                stats['unique_adrs'] = df_filtered['ADVERSE_EVENT'].nunique()
            elif table_name == "DRUG_INDICATIONS.csv":
                stats['unique_diseases'] = df_filtered['DRUG_INDICATION'].nunique()
        
        # Copy TB_DRUGS-specific analysis tables
        tb_drugs_tables = [
            "FAERS_TB_DRUGS_PROPORTIONATE_ANALYSIS.csv",
            "FAERS_TB_DRUGS_CONTINGENCY_TABLE.csv",
            "FAERS_TB_DRUGS_DRUG_ADVERSE_REACTIONS_COUNT.csv",
        ]
        
        for table_name in tb_drugs_tables:
            source_file = os.path.join(self.source_path, table_name)
            if os.path.exists(source_file):
                df = pd.read_csv(source_file, sep=FAERS_SEP, low_memory=False)
                df.columns = df.columns.str.upper()
                # Rename to standard names
                new_name = table_name.replace("FAERS_TB_DRUGS_", "")
                output_file = os.path.join(output_dir, new_name)
                df.to_csv(output_file, sep=FAERS_SEP, index=False)
                print(f"    ✓ Copied {len(df):,} records → {new_name}")
        
        print(f"\n  FAERS_TB_DRUGS Statistics:")
        print(f"    Case Reports: {stats.get('case_reports', 'N/A'):,}")
        print(f"    Unique Drugs: {stats.get('unique_drugs', 'N/A'):,}")
        print(f"    Unique ADRs:  {stats.get('unique_adrs', 'N/A'):,}")
        print(f"    Unique Diseases: {stats.get('unique_diseases', 'N/A'):,}")
        print(f"\n  ✓ FAERS_TB_DRUGS dataset saved to: {output_dir}\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Build FAERS dataset variants")
    parser.add_argument("--dataset", default="all",
                       choices=["all", "FAERS_ALL", "FAERS_TB", "FAERS_TB_DRUGS"],
                       help="Dataset to build (default: all)")
    parser.add_argument("--source-path", default=SOURCE_DATA_PATH,
                       help="Source data path")
    parser.add_argument("--output-path", default=OUTPUT_DATA_PATH,
                       help="Output data path")
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(args.source_path, args.output_path)
    
    print("=" * 70)
    print("FAERS DATASET BUILDER")
    print("=" * 70)
    print(f"Source: {args.source_path}")
    print(f"Output: {args.output_path}")
    print()
    
    all_stats = {}
    
    if args.dataset in ["all", "FAERS_ALL"]:
        all_stats["FAERS_ALL"] = builder.build_faers_all()
    
    if args.dataset in ["all", "FAERS_TB"]:
        all_stats["FAERS_TB"] = builder.build_faers_tb()
    
    if args.dataset in ["all", "FAERS_TB_DRUGS"]:
        all_stats["FAERS_TB_DRUGS"] = builder.build_faers_tb_drugs()
    
    # Print summary
    print("=" * 70)
    print("DATASET BUILD SUMMARY")
    print("=" * 70)
    print(f"{'Dataset':<20} {'Reports':>12} {'Drugs':>8} {'ADRs':>8} {'Diseases':>10}")
    print("-" * 70)
    for dataset, stats in all_stats.items():
        if stats:
            print(f"{dataset:<20} {stats.get('case_reports', 'N/A'):>12,} "
                  f"{stats.get('unique_drugs', 'N/A'):>8,} "
                  f"{stats.get('unique_adrs', 'N/A'):>8,} "
                  f"{stats.get('unique_diseases', 'N/A'):>10,}")
    print()
    print("✓ All datasets built successfully!")


if __name__ == "__main__":
    main()
