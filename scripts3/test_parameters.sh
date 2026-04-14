#!/bin/bash
# Parameter testing script for PreciseADR

# Activate environment
source adr_ehr_faers/bin/activate

# Base parameters (best from previous experiments)
ALPHA=0.0
TAU=0.05
SEED=42
VARIANT="xxx"

echo "======================================================================"
echo "PARAMETER TESTING SUITE"
echo "======================================================================"
echo ""

# Test 1: Learning Rate
echo "TEST 1: Learning Rate (lr)"
echo "----------------------------------------------------------------------"
for lr in 1e-3 5e-3 1e-2; do
    echo "Testing lr=${lr}..."
    python scripts3/step4_training.py \
        --variant ${VARIANT} \
        --alpha ${ALPHA} \
        --tau ${TAU} \
        --no-grid-search \
        --seed ${SEED} \
        --lr ${lr} \
        2>&1 | tee "output_faers_tb/param_test_lr_${lr}.log"
    echo ""
done

# Test 2: Dropout
echo "TEST 2: Dropout"
echo "----------------------------------------------------------------------"
for dropout in 0.1 0.3 0.5; do
    echo "Testing dropout=${dropout}..."
    python scripts3/step4_training.py \
        --variant ${VARIANT} \
        --alpha ${ALPHA} \
        --tau ${TAU} \
        --no-grid-search \
        --seed ${SEED} \
        --dropout ${dropout} \
        2>&1 | tee "output_faers_tb/param_test_dropout_${dropout}.log"
    echo ""
done

# Test 3: HGT Layers
echo "TEST 3: Number of HGT Layers"
echo "----------------------------------------------------------------------"
for layers in 2 3 4; do
    echo "Testing num_hgt_layers=${layers}..."
    python scripts3/step4_training.py \
        --variant ${VARIANT} \
        --alpha ${ALPHA} \
        --tau ${TAU} \
        --no-grid-search \
        --seed ${SEED} \
        --num-hgt-layers ${layers} \
        2>&1 | tee "output_faers_tb/param_test_layers_${layers}.log"
    echo ""
done

# Test 4: Attention Heads
echo "TEST 4: Number of Attention Heads"
echo "----------------------------------------------------------------------"
for heads in 2 4 8; do
    echo "Testing num_heads=${heads}..."
    python scripts3/step4_training.py \
        --variant ${VARIANT} \
        --alpha ${ALPHA} \
        --tau ${TAU} \
        --no-grid-search \
        --seed ${SEED} \
        --num-heads ${heads} \
        2>&1 | tee "output_faers_tb/param_test_heads_${heads}.log"
    echo ""
done

# Test 5: Embedding Dimension
echo "TEST 5: Embedding Dimension"
echo "----------------------------------------------------------------------"
for dim in 128 256 512; do
    echo "Testing embedding_dim=${dim}..."
    python scripts3/step4_training.py \
        --variant ${VARIANT} \
        --alpha ${ALPHA} \
        --tau ${TAU} \
        --no-grid-search \
        --seed ${SEED} \
        --embedding-dim ${dim} \
        2>&1 | tee "output_faers_tb/param_test_embdim_${dim}.log"
    echo ""
done

echo "======================================================================"
echo "PARAMETER TESTING COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to output_faers_tb/param_test_*.log"
echo ""
echo "To analyze results, run:"
echo "  python scripts3/analyze_param_tests.py"

