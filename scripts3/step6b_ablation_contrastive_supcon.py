"""
step6b_ablation_contrastive.py
================================
Ablation Category 2 (Section 6.2): Contrastive Learning Module

Research question: What is the contribution of the Patient Node
Augmentation Layer and the SupCon contrastive loss?

Variants:
  CL-1  Full PreciseADR (HGT + augmentation + SupCon)       — reference
  CL-2  Without augmentation (noise, but no dedicated Aug FC)
  CL-3  Without SupCon (α = 0, Focal Loss only)
  CL-4  Without both augmentation and SupCon

For CL-1 and CL-2 the full α × τ grid search is repeated.

Output:
  results/{variant}/ablation_contrastive.json
  results/{variant}/ablation_contrastive_grid_{cl1,cl2}.json   (5×5 heatmap)

Usage
-----
    python step6b_ablation_contrastive.py [--variant xxx]
"""

import os
import sys
import json
import argparse
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(__file__))
from config import GRAPH_PATH, MODEL_PATH, RESULTS_PATH, MODEL, ABLATION
from config import DATASET_NAME, OUTPUT_PATH
from step3_model_supcon import (PreciseADR, PreciseADRLoss, build_model_from_graph)
from step4_training_supcon import evaluate, evaluate_all_metrics, train_model, convert_to_serializable

METHOD = "supcon"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_supcon")

# ─────────────────────────────────────────────────────────────────────────────
# Variant factory
# ─────────────────────────────────────────────────────────────────────────────

class NoAugPreciseADR(PreciseADR):
    """CL-2: Replace augmentation with an identity (no dropout noise)."""
    def forward(self, x_dict, edge_index_dict, patient_mask=None):
        h = self.projection(x_dict)
        h = self.encoder(h, edge_index_dict)
        h_patient = h["patient"]
        if patient_mask is not None:
            h_patient = h_patient[patient_mask]
        # No augmentation: return h_patient for both z1 and z2 (identity)
        # SupCOn signature requires 4 return values: logits, h_orig, z1, z2 
        z1 = h_patient # No projection head applied (identity)
        z2 = h_patient # Same as z1 (no augmentation)
        logits = self.predictor(h_patient)
        return logits, h_patient, z1, z2


def build_variant(variant_id: str, train_graph, cfg: dict,
                   alpha: float, tau: float):
    """
    Build a model variant given the CL variant ID.
    alpha=0 disables SupCon in the loss (CL-3, CL-4).
    NoAugPreciseADR is used for CL-2 and CL-4.
    """
    base = build_model_from_graph(train_graph, cfg)

    if variant_id in ("CL-2", "CL-4"):
        # Swap augmentation with identity
        model = NoAugPreciseADR.__new__(NoAugPreciseADR)
        model.__dict__.update(base.__dict__)
    else:
        model = base

    loss_alpha = 0.0 if variant_id in ("CL-3", "CL-4") else alpha
    loss_fn    = PreciseADRLoss(alpha=loss_alpha, tau=tau,
                                 gamma=cfg["focal_gamma"])
    return model, loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# Grid search for CL-1 and CL-2
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_cl(variant_id: str, train_graph, val_graph,
                    cfg: dict, device: torch.device) -> tuple[float, float, dict]:
    """Returns (best_alpha, best_tau, heatmap_dict)."""
    print(f"    Grid search for {variant_id} …")
    heatmap = {}
    best_auc = -1.0
    best_a, best_t = cfg["alpha_range"][0], cfg["tau_range"][0]

    for alpha, tau in itertools.product(cfg["alpha_range"], cfg["tau_range"]):
        model, loss_fn = build_variant(variant_id, train_graph, cfg, alpha, tau)
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        opt = Adam(model.parameters(), lr=cfg["lr"])

        x_dict  = {nt: train_graph[nt].x.to(device) for nt in train_graph.node_types}
        ei_dict = {et: train_graph[et].edge_index.to(device)
                   for et in train_graph.edge_types
                   if hasattr(train_graph[et], "edge_index")}
        mask = train_graph["patient"].eval_mask.to(device)
        y    = train_graph["patient"].y.to(device)

        for ep in range(50):    # short grid-search training
            model.train(); opt.zero_grad()
            logits, h_orig, z1, z2 = model(x_dict, ei_dict, patient_mask=mask)
            loss, _, _ = loss_fn(logits, y[mask], h_orig, z1, z2)
            loss.backward(); opt.step()

        val_m = evaluate(model, val_graph, device)
        key   = f"a{alpha}_t{tau}"
        heatmap[key] = {"alpha": alpha, "tau": tau, "val_auc": val_m["AUC"]}

        if val_m["AUC"] > best_auc:
            best_auc = val_m["AUC"]; best_a = alpha; best_t = tau

    return best_a, best_t, heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Run all CL ablation variants
# ─────────────────────────────────────────────────────────────────────────────

def run_contrastive_ablation(variant: str, graph_path: str,
                              results_path: str, device: torch.device):
    print(f"\n{'='*70}")
    print(f"  ABLATION 6.2 – CONTRASTIVE LEARNING: {variant.upper()}")
    print(f"{'='*70}")

    graph_dir = os.path.join(graph_path, variant)
    out_dir   = os.path.join(results_path, variant)
    os.makedirs(out_dir, exist_ok=True)

    train_graph = torch.load(os.path.join(graph_dir, "train_graph.pt"), weights_only=False)
    val_graph   = torch.load(os.path.join(graph_dir, "val_graph.pt"), weights_only=False)
    test_graph  = torch.load(os.path.join(graph_dir, "test_graph.pt"), weights_only=False)

    cfg = dict(MODEL)
    metrics_path = os.path.join(out_dir, "final_metrics_supcon.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            prev = json.load(f)
        cfg["alpha"] = prev.get("alpha", cfg["alpha"])
        cfg["tau"]   = prev.get("tau",   cfg["tau"])

    all_results = {}

    for cl_id in ("CL-1", "CL-2", "CL-3", "CL-4"):
        print(f"\n  Running {cl_id} …")

        if cl_id in ("CL-1", "CL-2"):
            best_alpha, best_tau, heatmap = grid_search_cl(
                cl_id, train_graph, val_graph, cfg, device)
            # Save heatmap
            with open(os.path.join(out_dir, f"ablation_contrastive_grid_{cl_id.lower()}_supcon.json"), "w") as f:
                json.dump(convert_to_serializable(heatmap), f, indent=2)
        else:
            best_alpha = 0.0   # SupCon disabled
            best_tau   = cfg["tau"]

        # Full training
        model, loss_fn = build_variant(cl_id, train_graph, cfg,
                                        best_alpha, best_tau)
        model   = model.to(device)
        loss_fn = loss_fn.to(device)
        opt     = Adam(model.parameters(), lr=cfg["lr"])

        x_dict  = {nt: train_graph[nt].x.to(device) for nt in train_graph.node_types}
        ei_dict = {et: train_graph[et].edge_index.to(device)
                   for et in train_graph.edge_types
                   if hasattr(train_graph[et], "edge_index")}
        mask = train_graph["patient"].eval_mask.to(device)
        y    = train_graph["patient"].y.to(device)

        best_auc = -1.0; best_state = None; patience = cfg["patience"]
        for ep in range(cfg["max_epochs"]):
            model.train(); opt.zero_grad()
            logits, h_orig, z1, z2 = model(x_dict, ei_dict, patient_mask=mask)
            loss, _, _ = loss_fn(logits, y[mask], h_orig, z1, z2)
            loss.backward(); opt.step()

            if ep % 5 == 0:
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
        val_m  = evaluate(model, val_graph,  device)
        test_m = evaluate(model, test_graph, device)

        print(f"  {cl_id}: val_AUC={val_m['AUC']:.4f}  "
              f"test_AUC={test_m['AUC']:.4f}  "
              f"test_Hit@10={test_m.get('Hit@10', 0):.4f}")

        all_results[cl_id] = {
            "alpha": best_alpha, "tau": best_tau,
            "val_metrics": val_m, "test_metrics": test_m,
        }

    out_path = os.path.join(out_dir, "ablation_contrastive_supcon.json")
    with open(out_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Saved → {out_path}")
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Ablation 6.2: Contrastive Learning")
    p.add_argument("--graph-path",   default=GRAPH_PATH)
    p.add_argument("--results-path", default=RESULTS_PATH)
    p.add_argument("--model-path",   default=MODEL_PATH)
    p.add_argument("--variant",      default="all",
                   choices=["all", "xxx", "xxx_gender", "xxx_age"])
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",         default=123, type=int, help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args     = parse_args()
    device   = torch.device(args.device)
    variants = (["xxx", "xxx_gender", "xxx_age"]
                if args.variant == "all" else [args.variant])
    
    if args.seed is not None:
        from step4_training_supcon import set_seed
        set_seed(args.seed)
    
    for v in variants:
        if os.path.exists(os.path.join(args.graph_path, v)):
            run_contrastive_ablation(v, args.graph_path, args.results_path, device)
    print("\n✓ Ablation 6.2 (Contrastive Learning) complete!")


if __name__ == "__main__":
    main()
