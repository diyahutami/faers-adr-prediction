"""
Analyze ADR frequency distribution in FAERS-TB dataset

1. Load filtered ADVERSE_REACTIONS table
2. Compute frequency of each ADR
3. Identifies top-N most common ADRs
4. Visualizes distribution
5. Outputs recommendations for QC-3 filtering

Usage:
python analyze_adr_frequency.py --top 50
python analyze_adr_frequency.py --top 100 --save-plot
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import PREPROCESSED_PATH, FAERS_SEP, COL, DATASET_NAME

def load_adverse_reactions():
    """Load the filtered ADVERSE_REACTIONS table"""
    ar_path = os.path.join(PREPROCESSED_PATH, "ADVERSE_REACTIONS_FILTERED.csv")
    if not os.path.exists(ar_path):
        print(f"File not found: {ar_path}")
        print(" Run step1_preprocessing.py first!")
        sys.exit(1)
        
    print(f"Loading ADVERSE_REACTIONS from: {ar_path}")
    ar_df = pd.read_csv(ar_path, sep=FAERS_SEP, low_memory=False)
    print(f"Loaded {len(ar_df)} ADR records from ADVERSE_REACTIONS")
    return ar_df

def analyze_adr_frequency(ar_df, top_n=50):
    """
    Analyze ADR frequency distribution and identify top-N most common ADRs
    
    Returns:
    freq_adr: DataFrame with columns [ADR, count, cumulative_pct, rank]
    """
    
    adr_col = COL["adverse_event"]
    
    # Count occurrences
    freq = ar_df[adr_col].value_counts()
    total_records = len(ar_df)
    
    # Build analysis DataFrame
    freq_df = pd.DataFrame({
        "ADR": freq.index,
        "count": freq.values,
        "pct_of_records": (freq.values / total_records * 100).round(2)
    })

    # Add cumulative percentage
    freq_df["cumulative_pct"] = freq_df["pct_of_records"].cumsum().round(2)
    freq_df["rank"] = np.arange(1, len(freq_df) + 1)
    
    # Unique patients per ADR
    adr_patient_counts = ar_df.groupby(adr_col)[COL["primaryid"]].nunique()
    freq_df["unique_patients"] = freq_df["ADR"].map(adr_patient_counts)
    
    # Statistics
    print("\nADR Frequency Distribution Analysis")
    print(f"Total ADR records: {total_records}")
    print(f"Unique ADRs: {len(freq_df)}")
    print(f"Average records per ADR: {total_records / len(freq_df):.2f}")
    print(f"\nTop {min(top_n, len(freq_df))} ADRs:")
    top_df = freq_df.head(top_n)
    for _, row in top_df.iterrows():
        print(f"  #{row['rank']:3d}     {row['ADR'][:50]:50s}   "
              f"{row['count']:6,} records ({row['pct_of_records']:5.2f}%)   "
              f"{row['unique_patients']:5,} patients")
        
    # Coverage analysis
    print("\nCoverage Analysis:")
    for n in [10, 20, 50, 100, 150, 200]:
        if n <= len(freq_df):
            coverage = freq_df.head(n)["cumulative_pct"].iloc[-1]
            print(f"  Top {n:3d} ADRs cover {coverage:5.2f}% of all records")   

    # Long tail analysis
    print("\nLong Tail Analysis:")
    rare_adrs = freq_df[freq_df['count'] < 10]  # Define rare ADRs 
    print(f"  ADRs with <10 records: {len(rare_adrs):,} ({len(rare_adrs) / len(freq_df) * 100:.2f}%)")
    rare_adrs = freq_df[freq_df['count'] < 50]  # Define rare ADRs 
    print(f"  ADRs with <50 records: {len(rare_adrs):,} ({len(rare_adrs) / len(freq_df) * 100:.2f}%)")
    rare_adrs = freq_df[freq_df['count'] < 100]  # Define rare ADRs 
    print(f"  ADRs with <100 records: {len(rare_adrs):,} ({len(rare_adrs) / len(freq_df) * 100:.2f}%)")
    rare_adrs = freq_df[freq_df['count'] < 150]  # Define rare ADRs 
    print(f"  ADRs with <150 records: {len(rare_adrs):,} ({len(rare_adrs) / len(freq_df) * 100:.2f}%)")
    rare_adrs = freq_df[freq_df['count'] < 200]  # Define rare ADRs 
    print(f"  ADRs with <200 records: {len(rare_adrs):,} ({len(rare_adrs) / len(freq_df) * 100:.2f}%)")
    
    return freq_df

def plot_distribution(freq_df, top_n=50, save_path=None):
    """Plot ADR frequency distribution"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top N bar chart
    ax = axes[0, 0]
    top_data = freq_df.head(top_n)
    ax.barh(range(len(top_data)), top_data['count'], color='steelblue')
    ax.set_yticks(range(min(50, len(top_data)))) # Show only top 50 labels
    ax.set_yticklabels(top_data['ADR'].head(50).str[:40], fontsize=8)
    ax.set_xlabel("Number of Records")
    ax.set_title(f"Top {top_n} Most Frequent ADRs")
    ax.invert_yaxis()
    
    # Cumulative coverage
    ax = axes[0, 1]
    ax.plot(freq_df['rank'], freq_df['cumulative_pct'], color='darkgreen', linewidth=2)
    ax.axhline(y=70, color='blue', linestyle='--', label='70% Coverage')
    ax.axhline(y=80, color='red', linestyle='--', label='80% Coverage')
    ax.axhline(y=90, color='orange', linestyle='--', label='90% Coverage')
    ax.set_xlabel("ADR Rank")
    ax.set_ylabel("Cumulative Coverage (%)")
    ax.set_title("Cumulative ADR Coverage")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log-scale distribution
    ax = axes[1, 0]
    ax.loglog(freq_df['rank'], freq_df['count'], marker='.', linestyle='', alpha=0.6)
    ax.set_xlabel("ADR Rank (log scale)")
    ax.set_ylabel("Frequency(log scale)")
    ax.set_title("ADR Frequency Distribution (Log-Log Scale)")
    ax.grid(True, alpha=0.3)
    
    # Histogram of counts
    ax = axes[1, 1]
    ax.hist(freq_df['count'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel("Number of Records per ADR")
    ax.set_ylabel("Number of ADRs")
    ax.set_title("Distribution of ADR Frequencies")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
        
def save_top_adrs(freq_df, top_n, output_path):
    """Save top-N ADRs to file for use in preprocessing"""
    top_adrs = freq_df.head(top_n)['ADR'].tolist()
    
    # Save as text file (one per line)
    txt_path = output_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as f:
        for adr in top_adrs:
            f.write(f"{adr}\n")
    
    # Save as CSV with full details
    top_adrs_df = freq_df.head(top_n)
    top_adrs_df.to_csv(output_path, index=False)
    
    print(f"\n Top {top_n} ADRs saved to: ")
    print(f"  - Text file: {txt_path}")
    print(f"  - CSV file: {output_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Analyze ADR frequency distribution in FAERS-TB dataset")
    parser.add_argument("--top", type=int, default=50, help="Number of top ADRs to analyze (default: 50)")
    parser.add_argument("--save-plot", action='store_true', help="Save plot to file instead of displaying")
    parser.add_argument("--save-list",  action='store_true', help="Save list of top ADRs to file")
    parser.add_argument("--output-dir", default=PREPROCESSED_PATH, help="Output directory for saved files")
    args = parser.parse_args()
    
    # Load data
    ar_df = load_adverse_reactions()
    
    # Analyze frequency
    freq_df = analyze_adr_frequency(ar_df, top_n=args.top)
    
    # Plot
    if args.save_plot:
        plot_path = os.path.join(args.output_dir, f"adr_frequency_top_{args.top}.png")
        plot_distribution(freq_df, top_n=args.top, save_path=plot_path)
    else:
        plot_distribution(freq_df, top_n=args.top)
    
    # Save list
    if args.save_list:
        csv_path = os.path.join(args.output_dir, f"top_{args.top}_adrs.csv")
        save_top_adrs(freq_df, top_n=args.top, output_path=csv_path)

if __name__ == "__main__":
    main()
    