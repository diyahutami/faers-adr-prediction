"""
step6e_ablation_node_features.py
==================================
Ablation Category 5 (Section 6.5): Node Feature Ablation

Research question: Which patient node feature types are most important?

Variants (patient node features modified at graph-building time):
  Full features      : age_bin + gender + num_drugs + num_diseases   (baseline)
  No age             : gender + num_drugs + num_diseases
  No gender          : age_bin + num_drugs + num_diseases
  No demographics    : num_drugs + num_diseases
  No structural      : age_bin + gender  (no count features)
  No patient features: zero/random initialization

Drug and disease node features (one-hot) are held constant across variants.

Protocol: train PreciseADR (full model) for each variant; evaluate on
  - full test set
  - gender-related ADR subset
  - age-related ADR subset

Output:
  results/{variant}/ablation_node_features.json

Usage
-----
    python step6e_ablation_node_features.py [--variant xxx]
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import HeteroData

sys.path.insert(0, os.path.dirname(__file__))
from config import GRAPH_PATH, PREPROCESSED_PATH, MODEL_PATH, RESULTS_PATH, MODEL, ABLATION
from config import DATASET_NAME, OUTPUT_PATH
from step3_model_supcon import build_model_from_graph, PreciseADRLoss
from step4_training_supcon import evaluate_all_metrics, convert_to_serializable
from step6c_ablation_demographic_supcon import auc_per_adr, _load_related_adr_indices

METHOD = "supcon"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_supcon")

# ─────────────────────────────────────────────────────────────────────────────
# Patient feature masking
# ─────────────────────────────────────────────────────────────────────────────

# Patient feature layout (7 dims):
#   [0,1,2]  age one-hot (Youth, Adult, Elderly)
#   [3,4]    gender one-hot (Male, Female)
#   [5]      num_drugs_normed
#   [6]      num_diseases_normed

FEATURE_MASKS = {
    "Full"           : [0, 1, 2, 3, 4, 5, 6],
    "No_age"         : [3, 4, 5, 6],
    "No_gender"      : [0, 1, 2, 5, 6],
    "No_demographics": [5, 6],
    "No_structural"  : [0, 1, 2, 3, 4],
    "No_patient"     : [],   # zero vector
}


def _apply_feature_mask(x: torch.Tensor, indices: list) -> torch.Tensor:
    """Zero-out patient features not in indices."""
    if len(indices) == 0:
        return torch.zeros_like(x)
    mask = torch.zeros(x.shape[1], dtype=torch.bool)
    for i in indices:
        mask[i] = True
    x_mod       = x.clone()
    x_mod[:, ~mask] = 0.0
    return x_mod


def _masked_graph(graph: HeteroData, feature_indices: list) -> HeteroData:
    """Return a copy of graph with patient features masked."""
    g = graph  # shallow copy; we only replace patient features
    g["patient"].x = _apply_feature_mask(graph["patient"].x, feature_indices)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_eval(train_g: HeteroData, val_g: HeteroData, test_g: HeteroData,
                     cfg: dict, device: torch.device,
                     gender_idx: list, age_idx: list) -> dict:
    model   = build_model_from_graph(train_g, cfg).to(device)
    loss_fn = PreciseADRLoss(alpha=cfg["alpha"], tau=cfg["tau"],
                              gamma=cfg["focal_gamma"]).to(device)
    opt     = Adam(model.parameters(), lr=cfg["lr"])

    x_dict  = {nt: train_g[nt].x.to(device) for nt in train_g.node_types}
    ei_dict = {et: train_g[et].edge_index.to(device)
               for et in train_g.edge_types
               if hasattr(train_g[et], "edge_index")}
    mask = train_g["patient"].eval_mask.to(device)
    y    = train_g["patient"].y.to(device)

    best_auc = -1.0; best_state = None; patience = cfg["patience"]
    for ep in range(cfg["max_epochs"]):
        model.train(); opt.zero_grad()
        logits, h_orig, z1, z2 = model(x_dict, ei_dict, patient_mask=mask)
        loss, _, _ = loss_fn(logits, y[mask], h_orig, z1, z2)
        loss.backward(); opt.step()

        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                vx  = {nt: val_g[nt].x.to(device) for nt in val_g.node_types}
                vei = {et: val_g[et].edge_index.to(device)
                       for et in val_g.edge_types
                       if hasattr(val_g[et], "edge_index")}
                vm  = val_g["patient"].eval_mask.to(device)
                vl, _, _, _ = model(vx, vei, patient_mask=vm)
                vauc = torch.sigmoid(vl).cpu().numpy()
                vy   = val_g["patient"].y[val_g["patient"].eval_mask].numpy()
            v_auc = evaluate_all_metrics(vauc, vy)["AUC"]
            if v_auc > best_auc:
                best_auc   = v_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = cfg["patience"]
            else:
                patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        tx  = {nt: test_g[nt].x.to(device) for nt in test_g.node_types}
        tei = {et: test_g[et].edge_index.to(device)
               for et in test_g.edge_types
               if hasattr(test_g[et], "edge_index")}
        tm  = test_g["patient"].eval_mask.to(device)
        tl, _, _, _ = model(tx, tei, patient_mask=tm)
        scores   = torch.sigmoid(tl).cpu().numpy()
    y_true = test_g["patient"].y[test_g["patient"].eval_mask].numpy()

    full_m = evaluate_all_metrics(scores, y_true)
    g_auc  = auc_per_adr(scores, y_true, gender_idx)
    a_auc  = auc_per_adr(scores, y_true, age_idx)

    return {
        "test_metrics"        : full_m,
        "mean_gender_adr_auc" : float(np.nanmean(list(g_auc.values()))) if g_auc else 0.0,
        "mean_age_adr_auc"    : float(np.nanmean(list(a_auc.values()))) if a_auc else 0.0,
        "gender_adr_auc"      : {str(k): v for k, v in g_auc.items()},
        "age_adr_auc"         : {str(k): v for k, v in a_auc.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_node_feature_ablation(variant: str, graph_path: str,
                               results_path: str, device: torch.device):
    print(f"\n{'='*70}")
    print(f"  ABLATION 6.5 – NODE FEATURES: {variant.upper()}")
    print(f"{'='*70}")

    graph_dir = os.path.join(graph_path, variant)
    out_dir   = os.path.join(results_path, variant)
    os.makedirs(out_dir, exist_ok=True)

    full_train = torch.load(os.path.join(graph_dir, "train_graph.pt"), weights_only=False)
    full_val   = torch.load(os.path.join(graph_dir, "val_graph.pt"), weights_only=False)
    full_test  = torch.load(os.path.join(graph_dir, "test_graph.pt"), weights_only=False)

    cfg = dict(MODEL)
    metrics_p = os.path.join(out_dir, "final_metrics_supcon.json")
    if os.path.exists(metrics_p):
        with open(metrics_p) as f:
            prev = json.load(f)
        cfg["alpha"] = prev.get("alpha", cfg["alpha"])
        cfg["tau"]   = prev.get("tau",   cfg["tau"])

    # Load related ADRs
    adr_vocab = full_train.adr_vocab
    base_out  = os.path.dirname(results_path)
    gender_adrs_raw = []; age_adrs_raw = []
    gender_path = os.path.join(PREPROCESSED_PATH, "gender_related_adrs.json")
    age_path = os.path.join(PREPROCESSED_PATH, "age_related_adrs.json")
    
    if os.path.exists(gender_path):
        with open(gender_path) as f: 
            gender_adrs_raw = json.load(f)
    else:
        print(f"Warning: {gender_path} not found.")
        gender_adrs_raw = []
    
    if os.path.exists(age_path):
        with open(age_path) as f: 
            age_adrs_raw = json.load(f)
    else:
        print(f"Warning: {age_path} not found.")
        age_adrs_raw = []

    gender_idx = _load_related_adr_indices(adr_vocab, gender_adrs_raw,
                                            ABLATION.get("n_gender_adrs", 20))
    age_idx    = _load_related_adr_indices(adr_vocab, age_adrs_raw,
                                            ABLATION.get("n_age_adrs", 20))

    all_results = {}

    for feat_name, feat_cols in FEATURE_MASKS.items():
        print(f"\n  Variant: {feat_name}  (active dims: {feat_cols})")
        tr_g = _masked_graph(full_train, feat_cols)
        vl_g = _masked_graph(full_val,   feat_cols)
        te_g = _masked_graph(full_test,  feat_cols)

        res = _train_and_eval(tr_g, vl_g, te_g, cfg, device,
                              gender_idx, age_idx)
        print(f"    AUC={res['test_metrics']['AUC']:.4f}  "
              f"Hit@10={res['test_metrics'].get('Hit@10', 0):.4f}  "
              f"Gender-ADR_AUC={res['mean_gender_adr_auc']:.4f}  "
              f"Age-ADR_AUC={res['mean_age_adr_auc']:.4f}")
        all_results[feat_name] = res

    out_path = os.path.join(out_dir, "ablation_node_features_supcon.json")
    with open(out_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Saved → {out_path}")
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Ablation 6.5: Node Feature Ablation")
    p.add_argument("--graph-path",   default=GRAPH_PATH)
    p.add_argument("--results-path", default=RESULTS_PATH)
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
    for v in variants:
        if os.path.exists(os.path.join(args.graph_path, v)):
            run_node_feature_ablation(v, args.graph_path, args.results_path, device)
    print("\n✓ Ablation 6.5 (Node Feature Ablation) complete!")


if __name__ == "__main__":
    main()
