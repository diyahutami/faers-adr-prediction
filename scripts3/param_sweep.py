#!/usr/bin/env python3
"""
Parameter Sweep Script for PreciseADR
======================================
Tests different model hyperparameters systematically.

Usage:
    python scripts3/param_sweep.py --parameter lr --values 1e-3 5e-3 1e-2
    python scripts3/param_sweep.py --parameter dropout --values 0.1 0.3 0.5
    python scripts3/param_sweep.py --parameter num_hgt_layers --values 2 3 4
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import MODEL, GRAPH_PATH, MODEL_PATH, RESULTS_PATH

def run_experiment(param_name, param_value, variant="xxx", alpha=0.0, tau=0.05, seed=42):
    """Run a single training experiment with modified parameter."""
    
    # Create modified config
    config = dict(MODEL)
    config[param_name] = param_value
    config["alpha"] = alpha
    config["tau"] = tau
    
    # Save config to temporary file
    config_path = f"/tmp/config_{param_name}_{param_value}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training
    cmd = [
        "python", "scripts3/step4_training.py",
        "--variant", variant,
        "--alpha", str(alpha),
        "--tau", str(tau),
        "--seed", str(seed),
        "--no-grid-search"
    ]
    
    print(f"\n{'='*70}")
    print(f"  Testing {param_name} = {param_value}")
    print(f"{'='*70}\n")
    
    # Modify config temporarily
    original_config = dict(MODEL)
    MODEL.update(config)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Restore original config
    MODEL.update(original_config)
    
    # Parse results
    output = result.stdout
    test_auc = None
    hit10 = None
    
    for line in output.split('\n'):
        if 'AUC:' in line and test_auc is None:
            try:
                test_auc = float(line.split('AUC:')[1].split()[0])
            except:
                pass
        if 'Hit@10:' in line and hit10 is None:
            try:
                hit10 = float(line.split('Hit@10:')[1].split()[0])
            except:
                pass
    
    return {
        'param': param_name,
        'value': param_value,
        'test_auc': test_auc,
        'hit10': hit10,
        'output': output
    }

def main():
    parser = argparse.ArgumentParser(description="Parameter Sweep for PreciseADR")
    parser.add_argument('--parameter', required=True,
                       choices=['lr', 'dropout', 'num_hgt_layers', 'num_heads', 
                               'embedding_dim', 'focal_gamma'],
                       help='Parameter to test')
    parser.add_argument('--values', nargs='+', required=True,
                       help='Values to test')
    parser.add_argument('--variant', default='xxx')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--tau', type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Convert values to appropriate types
    param_values = []
    for v in args.values:
        try:
            # Try int first
            param_values.append(int(v))
        except:
            try:
                # Try float
                param_values.append(float(v))
            except:
                # Keep as string
                param_values.append(v)
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"  PARAMETER SWEEP: {args.parameter}")
    print(f"  Testing values: {param_values}")
    print(f"{'='*70}\n")
    
    for value in param_values:
        result = run_experiment(
            args.parameter, value,
            variant=args.variant,
            alpha=args.alpha,
            tau=args.tau,
            seed=args.seed
        )
        results.append(result)
        
        # Save intermediate results
        output_dir = Path("output_faers_tb/param_sweeps")
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / f"{args.parameter}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {args.parameter}")
    print(f"{'='*70}\n")
    print(f"{'Value':<15} {'Test AUC':<12} {'Hit@10':<12}")
    print('-' * 70)
    
    for r in results:
        auc_str = f"{r['test_auc']:.4f}" if r['test_auc'] else "N/A"
        hit_str = f"{r['hit10']:.4f}" if r['hit10'] else "N/A"
        print(f"{str(r['value']):<15} {auc_str:<12} {hit_str:<12}")
    
    # Find best
    valid_results = [r for r in results if r['test_auc'] is not None]
    if valid_results:
        best = max(valid_results, key=lambda x: x['test_auc'])
        print('\n' + '='*70)
        print(f"  BEST: {args.parameter} = {best['value']}")
        print(f"  Test AUC: {best['test_auc']:.4f}")
        print(f"  Hit@10:   {best['hit10']:.4f}")
        print('='*70)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()
