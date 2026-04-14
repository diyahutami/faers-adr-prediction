"""
step6f_ablation_summary.py
============================
Compile all ablation results into the summary table (Section 6.6).

Reads JSON output from steps 6a–6e and produces:
  - Console summary table
  - results/ablation_summary.csv  (full 6.6 table format)
  - results/ablation_summary.json

Usage
-----
    python step6f_ablation_summary.py
"""

import os
import sys
import json
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import RESULTS_PATH
from config import DATASET_NAME, OUTPUT_PATH

METHOD = "supcon"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_supcon")


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def compile_summary(results_path: str) -> pd.DataFrame:
    rows = []

    for variant in ["xxx", "xxx_gender", "xxx_age"]:
        out_dir = os.path.join(results_path, variant)

        # Main model metrics (from step4)
        main = _load_json(os.path.join(out_dir, "final_metrics_supcon.json"))
        tm   = main.get("test_metrics", {})

        # ── 6.1 Data Volume ──────────────────────────────────────────────
        dv = _load_json(os.path.join(out_dir, "ablation_data_volume_supcon.json"))
        for n_str, res in dv.items():
            rows.append({
                "ablation_category" : "6.1 Data Volume",
                "variant"           : variant,
                "config"            : f"{n_str} samples/class",
                "auc_mean"          : res.get("test_auc_mean", ""),
                "auc_std"           : res.get("test_auc_std", ""),
                "hit10_mean"        : res.get("test_hit10_mean", ""),
                "hit10_std"         : res.get("test_hit10_std", ""),
            })

        # ── 6.2 Contrastive ──────────────────────────────────────────────
        cl = _load_json(os.path.join(out_dir, "ablation_contrastive_supcon.json"))
        for cl_id, res in cl.items():
            m = res.get("test_metrics", {})
            rows.append({
                "ablation_category": "6.2 Contrastive",
                "variant"          : variant,
                "config"           : cl_id,
                "auc_mean"         : m.get("AUC", ""),
                "hit10_mean"       : m.get("Hit@10", ""),
            })

        # ── 6.3 Demographic ──────────────────────────────────────────────
        dem = _load_json(os.path.join(out_dir, "ablation_demographic_supcon.json"))
        for pert, res in dem.items():
            rows.append({
                "ablation_category"  : "6.3 Demographic",
                "variant"            : variant,
                "config"             : pert,
                "mean_gender_adr_auc": res.get("mean_gender_adr_auc", ""),
                "mean_age_adr_auc"   : res.get("mean_age_adr_auc", ""),
                "pct_red_gender"     : res.get("pct_reduction_gender", ""),
                "pct_red_age"        : res.get("pct_reduction_age", ""),
            })

        # ── 6.4 Graph Structure ───────────────────────────────────────────
        gs = _load_json(os.path.join(out_dir, "ablation_graph_structure_supcon.json"))
        for gname, m in gs.items():
            rows.append({
                "ablation_category": "6.4 Graph Structure",
                "variant"          : variant,
                "config"           : gname,
                "auc_mean"         : m.get("AUC", ""),
                "hit10_mean"       : m.get("Hit@10", ""),
            })

        # ── 6.5 Node Features ─────────────────────────────────────────────
        nf = _load_json(os.path.join(out_dir, "ablation_node_features_supcon.json"))
        for fname, res in nf.items():
            m = res.get("test_metrics", {})
            rows.append({
                "ablation_category"  : "6.5 Node Features",
                "variant"            : variant,
                "config"             : fname,
                "auc_mean"           : m.get("AUC", ""),
                "hit10_mean"         : m.get("Hit@10", ""),
                "mean_gender_adr_auc": res.get("mean_gender_adr_auc", ""),
                "mean_age_adr_auc"   : res.get("mean_age_adr_auc", ""),
            })

    df = pd.DataFrame(rows)
    return df


def main():
    p = argparse.ArgumentParser(description="Ablation Summary (Section 6.6)")
    p.add_argument("--results-path", default=RESULTS_PATH)
    args = p.parse_args()

    df = compile_summary(args.results_path)

    if df.empty:
        print("No ablation results found yet. Run steps 6a–6e first.")
        return

    # Console print
    print("\n" + "="*90)
    print("  ABLATION STUDY SUMMARY TABLE (Section 6.6)")
    print("="*90)
    print(df.to_string(index=False))

    # Save
    csv_path  = os.path.join(args.results_path, "ablation_summary_supcon.csv")
    json_path = os.path.join(args.results_path, "ablation_summary_supcon.json")
    os.makedirs(args.results_path, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    print(f"\n  Saved → {csv_path}")
    print(f"  Saved → {json_path}")
    print("\n✓ Ablation Summary (6.6) complete!")


if __name__ == "__main__":
    main()
