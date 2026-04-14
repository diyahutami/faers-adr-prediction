#!/usr/bin/env python3
"""
Analyze Parameter Testing Results
==================================
Parse all parameter test logs and create comparison tables.

Usage:
    python scripts3/analyze_param_tests.py
"""

import re
import os
from pathlib import Path
from collections import defaultdict

def extract_metrics(log_file):
    """Extract test AUC and Hit@10 from log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Find test metrics section
        test_section = content.split('Test metrics:')
        if len(test_section) < 2:
            return None, None
        
        metrics = test_section[1]
        
        # Extract AUC
        auc_match = re.search(r'AUC:\s+([\d.]+)', metrics)
        auc = float(auc_match.group(1)) if auc_match else None
        
        # Extract Hit@10
        hit10_match = re.search(r'Hit@10:\s+([\d.]+)', metrics)
        hit10 = float(hit10_match.group(1)) if hit10_match else None
        
        return auc, hit10
    except:
        return None, None

def analyze_parameter(param_name, log_pattern, results_dir):
    """Analyze a specific parameter's results."""
    results = []
    
    # Find all log files for this parameter
    log_files = sorted(Path(results_dir).glob(log_pattern))
    
    for log_file in log_files:
        # Extract parameter value from filename
        filename = log_file.stem
        value_match = re.search(r'_(\d+\.?\d*)\.log', str(log_file))
        if value_match:
            value = value_match.group(1)
        else:
            value = filename.split('_')[-1]
        
        # Extract metrics
        auc, hit10 = extract_metrics(log_file)
        
        if auc is not None:
            results.append({
                'value': value,
                'auc': auc,
                'hit10': hit10,
                'log_file': str(log_file)
            })
    
    return results

def main():
    results_dir = Path("output_faers_tb/param_tests")
    
    if not results_dir.exists():
        print("Error: No test results found in output_faers_tb/param_tests/")
        print("Run the test scripts first!")
        return
    
    print("=" * 80)
    print("PARAMETER TESTING RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Define parameters to analyze
    params = {
        'Learning Rate': 'lr_*.log',
        'Dropout': 'dropout_*.log',
        'HGT Layers': 'hgt_layers_*.log',
        'Attention Heads': 'attention_heads_*.log',
        'Embedding Dimension': 'embedding_dim_*.log',
        'Focal Gamma': 'focal_gamma_*.log',
        'Batch Size': 'batch_size_*.log'
    }
    
    all_results = {}
    
    for param_name, pattern in params.items():
        results = analyze_parameter(param_name, pattern, results_dir)
        
        if results:
            all_results[param_name] = results
            
            print(f"\n{param_name}")
            print("-" * 80)
            print(f"{'Value':<20} {'Test AUC':<15} {'Hit@10':<15} {'Rank':<10}")
            print("-" * 80)
            
            # Sort by AUC descending
            sorted_results = sorted(results, key=lambda x: x['auc'], reverse=True)
            
            for rank, r in enumerate(sorted_results, 1):
                marker = "⭐ BEST" if rank == 1 else ""
                print(f"{r['value']:<20} {r['auc']:<15.4f} {r['hit10']:<15.4f} {rank:<10} {marker}")
            
            # Find best
            best = sorted_results[0]
            print(f"\n✅ Best {param_name}: {best['value']}")
            print(f"   Test AUC: {best['auc']:.4f}, Hit@10: {best['hit10']:.4f}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 80)
    print()
    
    if all_results:
        for param_name, results in all_results.items():
            if results:
                best = max(results, key=lambda x: x['auc'])
                print(f"  {param_name:<25}: {best['value']}")
        
        print()
        print("To use these parameters, update scripts3/config.py:")
        print()
        print("MODEL = {")
        for param_name, results in all_results.items():
            if results:
                best = max(results, key=lambda x: x['auc'])
                if param_name == 'Learning Rate':
                    print(f"    \"lr\": {best['value']},")
                elif param_name == 'Dropout':
                    print(f"    \"dropout\": {best['value']},")
                elif param_name == 'HGT Layers':
                    print(f"    \"num_hgt_layers\": {best['value']},")
                elif param_name == 'Attention Heads':
                    print(f"    \"num_heads\": {best['value']},")
                elif param_name == 'Embedding Dimension':
                    print(f"    \"embedding_dim\": {best['value']},")
                elif param_name == 'Focal Gamma':
                    print(f"    \"focal_gamma\": {best['value']},")
                elif param_name == 'Batch Size':
                    print(f"    \"batch_size\": {best['value']},")
        print("    # ... other parameters")
        print("}")
    else:
        print("No test results found!")
        print("Run the test scripts first:")
        print("  ./test_lr_example.sh")
        print("  ./test_dropout.sh")
        print("  ./test_hgt_layers.sh")
        print("  ./test_attention_heads.sh")
        print("  ./test_embedding_dim.sh")
        print("  ./test_focal_gamma.sh")
        print("  ./test_batch_size.sh")
    
    print()
    print("=" * 80)

if __name__ == '__main__':
    main()

