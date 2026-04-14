#!/bin/bash
#
# run_all_datasets.sh
# ===================
# Run the full pipeline for all three dataset variants:
# FAERS_ALL, FAERS_TB, and FAERS_TB_DRUGS
#
# Usage:
#   ./run_all_datasets.sh                    # Run all steps for all datasets
#   ./run_all_datasets.sh --steps 1 2       # Run only steps 1 and 2
#   ./run_all_datasets.sh --variant xxx     # Run only XXX variant

set -e  # Exit on error

echo "========================================================================"
echo "RUNNING PIPELINE FOR ALL THREE DATASETS"
echo "========================================================================"
echo ""

# Parse arguments
ARGS="$@"

# ── Run FAERS_ALL ─────────────────────────────────────────────────────────
echo "========================================================================"
echo "1/3: Processing FAERS_ALL (Full FAERS Dataset)"
echo "========================================================================"
echo ""
DATASET_NAME=FAERS_ALL python run_pipeline.py $ARGS
echo ""
echo "✓ FAERS_ALL processing complete!"
echo ""

# ── Run FAERS_TB ──────────────────────────────────────────────────────────
echo "========================================================================"
echo "2/3: Processing FAERS_TB (Tuberculosis Cases)"
echo "========================================================================"
echo ""
DATASET_NAME=FAERS_TB python run_pipeline.py $ARGS
echo ""
echo "✓ FAERS_TB processing complete!"
echo ""

# ── Run FAERS_TB_DRUGS ────────────────────────────────────────────────────
echo "========================================================================"
echo "3/3: Processing FAERS_TB_DRUGS (TB Cases with TB-Specific Drugs)"
echo "========================================================================"
echo ""
DATASET_NAME=FAERS_TB_DRUGS python run_pipeline.py $ARGS
echo ""
echo "✓ FAERS_TB_DRUGS processing complete!"
echo ""

# ── Summary ───────────────────────────────────────────────────────────────
echo "========================================================================"
echo "ALL DATASETS PROCESSED SUCCESSFULLY!"
echo "========================================================================"
echo ""
echo "Results are stored in:"
echo "  - output_faers_all/results_FAERS_ALL/"
echo "  - output_faers_tb/results_FAERS_TB/"
echo "  - output_faers_tb_drugs/results_FAERS_TB_DRUGS/"
echo ""
echo "To compare results, check the respective results folders."
echo ""
