"""
run_pipeline.py
================
Master runner for the full PreciseADR reproduction pipeline - scripts3 version.

SCRIPTS3 UPDATE: Multi-dataset support for FAERS_ALL, FAERS_TB, FAERS_TB_DRUGS
  - Use DATASET_NAME environment variable to select dataset
  - Or modify DATASET_NAME in config.py
  - All outputs are dataset-specific (separate folders)

Runs all steps in sequence:
  Step 1: Preprocessing
  Step 2: AER Graph Construction
  Step 3: (model definition – no CLI entry)
  Step 4: Training + Grid Search
  Step 5: Baseline Evaluation
  Step 6a–6f: Ablation Studies

Usage
-----
    # Using default dataset from config.py
    python run_pipeline.py                     # full pipeline
    python run_pipeline.py --steps 1 2        # only steps 1 and 2
    python run_pipeline.py --variant xxx      # only XXX dataset variant
    python run_pipeline.py --device cuda      # GPU training
    
    # Selecting specific dataset
    DATASET_NAME=FAERS_TB python run_pipeline.py
    DATASET_NAME=FAERS_TB_DRUGS python run_pipeline.py --steps 1 2 4
"""

import sys
import argparse
import torch

from config import (
    DATA_PATH, OUTPUT_PATH, GRAPH_PATH, MODEL_PATH, RESULTS_PATH,
    DATASET_NAME, print_config
)

def main():
    p = argparse.ArgumentParser(description="PreciseADR Full Pipeline")
    p.add_argument("--steps",    nargs="*", type=str, default=None,
                   help="Steps to run, e.g. --steps 1 2 4. Default: all.")
    p.add_argument("--variant",  default="all",
                   choices=["all", "xxx", "xxx_gender", "xxx_age"])
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--drugbank-mapping", default=None)
    p.add_argument("--meddra-file",      default=None)
    p.add_argument("--no-grid-search",   action="store_true")
    args = p.parse_args()

    steps = set(args.steps) if args.steps else {
        "1", "2", "4", "5", "6a", "6b", "6c", "6d", "6e", "6f"
    }

    device = torch.device(args.device)
    
    # Print configuration
    print_config()
    print(f"\nDevice: {device}")
    print(f"Steps:  {sorted(steps)}\n")

    # ── Step 1 ───────────────────────────────────────────────────────────────
    if "1" in steps:
        from step1_preprocessing import Step1Preprocessor
        proc = Step1Preprocessor(DATA_PATH, OUTPUT_PATH)
        proc.load_data()
        if args.drugbank_mapping:
            proc.add_drugbank_mapping(args.drugbank_mapping)
        if args.meddra_file:
            proc.add_meddra_mapping(args.meddra_file)
        proc.apply_quality_control()
        proc.apply_drug_interference_filtering()
        proc.print_statistics_table()
        proc.identify_gender_related_adrs()
        proc.identify_age_related_adrs()
        proc.build_specialized_datasets()
        proc.split_datasets()
        proc.save()

    # ── Step 2 ───────────────────────────────────────────────────────────────
    if "2" in steps:
        from step2_aer_graph_construction import AERGraphBuilder
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        AERGraphBuilder(DATA_PATH, OUTPUT_PATH, GRAPH_PATH).build_all(variants)

    # ── Step 4 ───────────────────────────────────────────────────────────────
    if "4" in steps:
        from scripts3.step4_training_supcon import train_variant
        from config import MODEL
        import os
        cfg = dict(MODEL)
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        for v in variants:
            if os.path.exists(os.path.join(GRAPH_PATH, v)):
                train_variant(v, cfg, GRAPH_PATH, MODEL_PATH, RESULTS_PATH,
                              do_grid_search=not args.no_grid_search,
                              device=device)

    # ── Step 5 ───────────────────────────────────────────────────────────────
    if "5" in steps:
        from scripts3.step5_baselines_supcon import run_baselines
        import os
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        for v in variants:
            if os.path.exists(os.path.join(GRAPH_PATH, v)):
                run_baselines(v, GRAPH_PATH, RESULTS_PATH, device)

    # ── Step 6a ──────────────────────────────────────────────────────────────
    if "6a" in steps:
        from step6a_ablation_data_volume import run_data_volume_ablation
        import os
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        for v in variants:
            if os.path.exists(os.path.join(GRAPH_PATH, v)):
                run_data_volume_ablation(v, GRAPH_PATH, RESULTS_PATH, MODEL_PATH, device)

    # ── Step 6b ──────────────────────────────────────────────────────────────
    if "6b" in steps:
        from step6b_ablation_contrastive import run_contrastive_ablation
        import os
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        for v in variants:
            if os.path.exists(os.path.join(GRAPH_PATH, v)):
                run_contrastive_ablation(v, GRAPH_PATH, RESULTS_PATH, device)

    # ── Step 6c ──────────────────────────────────────────────────────────────
    if "6c" in steps:
        from step6c_ablation_demographic import run_demographic_ablation
        import os
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        for v in variants:
            if os.path.exists(os.path.join(GRAPH_PATH, v)):
                run_demographic_ablation(v, GRAPH_PATH, RESULTS_PATH, MODEL_PATH, device)

    # ── Step 6d ──────────────────────────────────────────────────────────────
    if "6d" in steps:
        from step6d_ablation_graph_structure import run_graph_structure_ablation
        import os
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        for v in variants:
            if os.path.exists(os.path.join(GRAPH_PATH, v)):
                run_graph_structure_ablation(v, GRAPH_PATH, RESULTS_PATH, device)

    # ── Step 6e ──────────────────────────────────────────────────────────────
    if "6e" in steps:
        from step6e_ablation_node_features import run_node_feature_ablation
        import os
        variants = (["xxx", "xxx_gender", "xxx_age"]
                    if args.variant == "all" else [args.variant])
        for v in variants:
            if os.path.exists(os.path.join(GRAPH_PATH, v)):
                run_node_feature_ablation(v, GRAPH_PATH, RESULTS_PATH, device)

    # ── Step 6f ──────────────────────────────────────────────────────────────
    if "6f" in steps:
        from step6f_ablation_summary import compile_summary
        df = compile_summary(RESULTS_PATH)
        if not df.empty:
            import os
            df.to_csv(os.path.join(RESULTS_PATH, "ablation_summary.csv"), index=False)
            print(df.to_string(index=False))

    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
