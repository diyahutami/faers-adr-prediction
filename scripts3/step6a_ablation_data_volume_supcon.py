"""
step6a_ablation_data_volume.py
================================
Ablation Category 1 (Section 6.1): Data Volume Scaling

Research question: How does the volume of labelled training data
affect PreciseADR's predictive performance?

Protocol:
  For each configuration in {10, 20, 30, 50, 70, 100} samples per ADR class:
    1. Randomly sample the specified number of patients per ADR class
       from the training set (stratified, fixed seed).
    2. Train full PreciseADR (HGT + contrastive learning, optimal α/τ).
    3. Evaluate on the full validation and test sets (unchanged).
    4. Repeat with 3 different random seeds; report mean ± std.

Output:
  results/{variant}/ablation_data_volume.json
  results/{variant}/ablation_data_volume_summary.csv

Usage
-----
    python step6a_ablation_data_volume.py [--variant xxx|xxx_gender|xxx_age]
"""

import os
import sys
import json
import argparse
import copy

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(__file__))
from config import GRAPH_PATH, MODEL_PATH, RESULTS_PATH, MODEL, ABLATION, EVAL
from config import DATASET_NAME, OUTPUT_PATH
from step3_model_supcon import build_model_from_graph, PreciseADRLoss
from step4_training_supcon import evaluate, convert_to_serializable

METHOD = "supcon"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_supcon")
# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def sample_patients_per_class(train_graph, n_per_class: int,
                               rng: np.random.Generator) -> torch.Tensor:
    """
    Return a boolean mask over train graph patients selecting at most
    n_per_class patients per ADR class.
    """
    mask = train_graph["patient"].eval_mask
    y    = train_graph["patient"].y[mask].numpy()   # (N_train, N_adr)
    n_train, n_adr = y.shape

    selected = set()
    for adr_idx in range(n_adr):
        pos = np.where(y[:, adr_idx] > 0)[0]
        if len(pos) == 0:
            continue
        k = min(n_per_class, len(pos))
        chosen = rng.choice(pos, size=k, replace=False)
        selected.update(chosen.tolist())

    # Build a full mask aligned with the graph's patient node list
    full_mask = torch.zeros(train_graph["patient"].x.shape[0], dtype=torch.bool)
    eval_idx  = torch.where(mask)[0]
    for local in selected:
        full_mask[eval_idx[local]] = True
    return full_mask


# ─────────────────────────────────────────────────────────────────────────────
# Training loop for one ablation run
# ─────────────────────────────────────────────────────────────────────────────

def train_with_subset_mask(train_graph, val_graph, sampled_mask,
                            cfg: dict, device: torch.device) -> dict:
    """Train PreciseADR using only the patients indicated by sampled_mask."""
    model   = build_model_from_graph(train_graph, cfg).to(device)
    loss_fn = PreciseADRLoss(alpha=cfg["alpha"], tau=cfg["tau"],
                              gamma=cfg["focal_gamma"]).to(device)
    opt     = Adam(model.parameters(), lr=cfg["lr"])

    x_dict  = {nt: train_graph[nt].x.to(device) for nt in train_graph.node_types}
    ei_dict = {et: train_graph[et].edge_index.to(device)
               for et in train_graph.edge_types
               if hasattr(train_graph[et], "edge_index")}
    y_sub   = train_graph["patient"].y[sampled_mask].to(device)

    best_auc = -1.0; best_state = None; patience = cfg["patience"]

    for epoch in range(cfg["max_epochs"]):
        model.train()
        opt.zero_grad()
        logits, h_orig, z1, z2 = model(x_dict, ei_dict, patient_mask=sampled_mask.to(device))
        loss, _, _ = loss_fn(logits, y_sub, h_orig, z1, z2)
        loss.backward(); opt.step()

        if epoch % 5 == 0 or epoch == cfg["max_epochs"] - 1:
            val_m = evaluate(model, val_graph, device)
            if val_m["AUC"] > best_auc:
                best_auc   = val_m["AUC"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = cfg["patience"]
            else:
                patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_state)
    return evaluate(model, val_graph, device), evaluate(model, torch.load(
        # Reload test graph path from graph_path – passed via cfg
        os.path.join(cfg["_graph_dir"], "test_graph.pt"), weights_only=False), device)


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_data_volume_ablation(variant: str, graph_path: str,
                              results_path: str, model_path: str,
                              device: torch.device):
    print(f"\n{'='*70}")
    print(f"  ABLATION 6.1 – DATA VOLUME SCALING: {variant.upper()}")
    print(f"{'='*70}")

    graph_dir = os.path.join(graph_path, variant)
    out_dir   = os.path.join(results_path, variant)
    os.makedirs(out_dir, exist_ok=True)

    train_graph = torch.load(os.path.join(graph_dir, "train_graph.pt"), weights_only=False)
    val_graph   = torch.load(os.path.join(graph_dir, "val_graph.pt"), weights_only=False)

    # Load best α/τ from training results if available
    metrics_path = os.path.join(results_path, variant, "final_metrics_supcon.json")
    cfg = dict(MODEL)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            prev = json.load(f)
        cfg["alpha"] = prev.get("alpha", cfg["alpha"])
        cfg["tau"]   = prev.get("tau",   cfg["tau"])
    cfg["_graph_dir"] = graph_dir   # pass path through for test graph reload

    samples_per_class = ABLATION["samples_per_class"]
    n_seeds           = ABLATION["n_seeds"]

    all_results = {}

    for n in samples_per_class:
        print(f"\n  n_per_class = {n}")
        seed_results = []

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed + 42)
            sampled_mask = sample_patients_per_class(train_graph, n, rng)
            n_sampled    = int(sampled_mask.sum())
            print(f"    seed={seed}  sampled={n_sampled:,} patients")

            val_m, test_m = train_with_subset_mask(
                train_graph, val_graph, sampled_mask, cfg, device)
            seed_results.append({"seed": seed, "n_sampled": n_sampled,
                                  "val": val_m, "test": test_m})
            print(f"    → val_AUC={val_m['AUC']:.4f}  "
                  f"test_AUC={test_m['AUC']:.4f}  "
                  f"test_Hit@10={test_m.get('Hit@10', 0):.4f}")

        # Aggregate over seeds
        test_aucs   = [r["test"]["AUC"]         for r in seed_results]
        test_hit10  = [r["test"].get("Hit@10", 0) for r in seed_results]
        all_results[str(n)] = {
            "seeds"         : seed_results,
            "test_auc_mean" : float(np.mean(test_aucs)),
            "test_auc_std"  : float(np.std(test_aucs)),
            "test_hit10_mean": float(np.mean(test_hit10)),
            "test_hit10_std" : float(np.std(test_hit10)),
        }
        print(f"  → AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}  "
              f"Hit@10: {np.mean(test_hit10):.4f} ± {np.std(test_hit10):.4f}")

    # Save JSON
    out_json = os.path.join(out_dir, "ablation_data_volume_supcon.json")
    with open(out_json, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Saved → {out_json}")

    # Save summary CSV (learning curve table)
    rows = []
    for n, res in all_results.items():
        rows.append({
            "samples_per_class" : int(n),
            "test_auc_mean"     : res["test_auc_mean"],
            "test_auc_std"      : res["test_auc_std"],
            "test_hit10_mean"   : res["test_hit10_mean"],
            "test_hit10_std"    : res["test_hit10_std"],
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "ablation_data_volume_summary_supcon.csv"), index=False)
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Ablation 6.1: Data Volume Scaling")
    p.add_argument("--graph-path",   default=GRAPH_PATH)
    p.add_argument("--results-path", default=RESULTS_PATH)
    p.add_argument("--model-path",   default=MODEL_PATH)
    p.add_argument("--variant",      default="all",
                   choices=["all", "xxx", "xxx_gender", "xxx_age"])
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args     = parse_args()
    device   = torch.device(args.device)
    variants = (["xxx", "xxx_gender", "xxx_age"]
                if args.variant == "all" else [args.variant])
    for v in variants:
        if os.path.exists(os.path.join(args.graph_path, v)):
            run_data_volume_ablation(v, args.graph_path, args.results_path,
                                     args.model_path, device)
    print("\n✓ Ablation 6.1 (Data Volume Scaling) complete!")


if __name__ == "__main__":
    main()
