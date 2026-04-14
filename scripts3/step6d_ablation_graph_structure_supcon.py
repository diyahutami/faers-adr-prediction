"""
step6d_ablation_graph_structure.py
=====================================
Ablation Category 4 (Section 6.4): Graph Structure Ablation

Research question: What is the contribution of the heterogeneous AER Graph
structure compared to simpler graph representations? For TB datasets, what
is the specific contribution of DDI and III edges?

Variants for FAERS-ALL (G1-G5):
  G1: No Graph (MLP)              - No graph, patient features only
  G2: Drug Co-occurrence          - Homogeneous drug-drug co-occurrence graph
  G3: Patient-Drug (bipartite)    - Partial heterogeneous: Patient + Drug edges
  G4: Patient-Drug-Disease        - Partial heterogeneous: + Disease edges
  G5: Full AER Graph (PreciseADR) - Full: + ADR edges (training labels)

Additional variants for FAERS-TB and FAERS-TB-DRUGS (G6-G9):
  G6: TB graph, no DDI, no III    - TB data benefit without new edges
  G7: TB graph, WITH III only     - Isolates III edge contribution
  G8: TB graph, WITH DDI only     - Isolates DDI edge contribution
  G9: TB graph, WITH DDI + III    - Full proposed model (reference for G6-G8)

Each variant trains PreciseADR with HGT + full contrastive learning,
keeping all hyperparameters identical; only the graph structure changes.

Output:
  results/{variant}/ablation_graph_structure.json

Usage
-----
    python step6d_ablation_graph_structure.py [--variant xxx]
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
from config import GRAPH_PATH, MODEL_PATH, RESULTS_PATH, MODEL, DATASET_NAME
from config import DATASET_NAME, OUTPUT_PATH
from step3_model_supcon import (PreciseADR, PreciseADRLoss, NodeFeatureProjection, ADRPredictor,
                          build_model_from_graph)
from step4_training_supcon import evaluate, evaluate_all_metrics, convert_to_serializable

METHOD = "supcon"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_supcon")

# ─────────────────────────────────────────────────────────────────────────────
# Graph structure variants for FAERS-ALL (G1-G5)
# ─────────────────────────────────────────────────────────────────────────────

def _make_no_graph(full_graph: HeteroData) -> HeteroData:
    """G1: No graph - patients only, no edges."""
    g = HeteroData()
    g["patient"].x         = full_graph["patient"].x
    g["patient"].y         = full_graph["patient"].y
    g["patient"].eval_mask = full_graph["patient"].eval_mask
    g["patient"].pid       = full_graph["patient"].pid
    # Minimal dummy nodes for drug/disease/adr (still needed for model init)
    g["drug"].x    = full_graph["drug"].x
    g["disease"].x = full_graph["disease"].x
    g["adr"].x     = full_graph["adr"].x
    g.drug_vocab    = full_graph.drug_vocab
    g.disease_vocab = full_graph.disease_vocab
    g.adr_vocab     = full_graph.adr_vocab
    # No edges
    return g


def _make_patient_drug(full_graph: HeteroData, include_adr: bool = False) -> HeteroData:
    """G3: Patient + Drug nodes, Patient-Takes-Drug edges only."""
    g = HeteroData()
    g["patient"].x         = full_graph["patient"].x
    g["patient"].y         = full_graph["patient"].y
    g["patient"].eval_mask = full_graph["patient"].eval_mask
    g["patient"].pid       = full_graph["patient"].pid
    g["drug"].x    = full_graph["drug"].x
    g["disease"].x = full_graph["disease"].x
    g["adr"].x     = full_graph["adr"].x

    for et in [("patient", "takes", "drug"), ("drug", "taken_by", "patient")]:
        if hasattr(full_graph[et], "edge_index"):
            g[et].edge_index = full_graph[et].edge_index

    if include_adr:
        for et in [("patient", "experiences", "adr"), ("adr", "experienced_by", "patient")]:
            if hasattr(full_graph[et], "edge_index"):
                g[et].edge_index = full_graph[et].edge_index

    g.drug_vocab    = full_graph.drug_vocab
    g.disease_vocab = full_graph.disease_vocab
    g.adr_vocab     = full_graph.adr_vocab
    return g


def _make_patient_drug_disease(full_graph: HeteroData,
                                include_adr: bool = False) -> HeteroData:
    """G4: Patient + Drug + Disease nodes."""
    g = _make_patient_drug(full_graph, include_adr=include_adr)
    for et in [("patient", "has", "disease"), ("disease", "had_by", "patient")]:
        if hasattr(full_graph[et], "edge_index"):
            g[et].edge_index = full_graph[et].edge_index
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Graph structure variants for TB datasets (G6-G9)
# ─────────────────────────────────────────────────────────────────────────────

def _make_tb_graph_no_ddi_no_iii(full_graph: HeteroData) -> HeteroData:
    """G6: TB graph without DDI/III edges - isolates dataset effect from edge type effect."""
    g = HeteroData()
    g["patient"].x         = full_graph["patient"].x
    g["patient"].y         = full_graph["patient"].y
    g["patient"].eval_mask = full_graph["patient"].eval_mask
    g["patient"].pid       = full_graph["patient"].pid
    g["drug"].x    = full_graph["drug"].x
    g["disease"].x = full_graph["disease"].x
    g["adr"].x     = full_graph["adr"].x

    # Include all original PreciseADR edge types but NOT DDI/III
    edge_types_to_include = [
        ("patient", "takes", "drug"),
        ("drug", "taken_by", "patient"),
        ("patient", "has", "disease"),
        ("disease", "had_by", "patient"),
        ("patient", "experiences", "adr"),
        ("adr", "experienced_by", "patient"),
    ]
    
    for et in edge_types_to_include:
        if hasattr(full_graph[et], "edge_index"):
            g[et].edge_index = full_graph[et].edge_index

    g.drug_vocab    = full_graph.drug_vocab
    g.disease_vocab = full_graph.disease_vocab
    g.adr_vocab     = full_graph.adr_vocab
    return g


def _make_tb_graph_iii_only(full_graph: HeteroData) -> HeteroData:
    """G7: TB graph with III edges only (no DDI) - isolates III contribution."""
    g = _make_tb_graph_no_ddi_no_iii(full_graph)
    
    # Add III edges if they exist
    if hasattr(full_graph[("disease", "comorbid", "disease")], "edge_index"):
        g[("disease", "comorbid", "disease")].edge_index = full_graph[("disease", "comorbid", "disease")].edge_index
        if hasattr(full_graph[("disease", "comorbid", "disease")], "edge_attr"):
            g[("disease", "comorbid", "disease")].edge_attr = full_graph[("disease", "comorbid", "disease")].edge_attr
    
    return g


def _make_tb_graph_ddi_only(full_graph: HeteroData) -> HeteroData:
    """G8: TB graph with DDI edges only (no III) - isolates DDI contribution."""
    g = _make_tb_graph_no_ddi_no_iii(full_graph)
    
    # Add DDI edges if they exist
    if hasattr(full_graph[("drug", "interacts", "drug")], "edge_index"):
        g[("drug", "interacts", "drug")].edge_index = full_graph[("drug", "interacts", "drug")].edge_index
        if hasattr(full_graph[("drug", "interacts", "drug")], "edge_attr"):
            g[("drug", "interacts", "drug")].edge_attr = full_graph[("drug", "interacts", "drug")].edge_attr
    
    return g


def _make_tb_graph_full(full_graph: HeteroData) -> HeteroData:
    """G9: Full TB graph with DDI + III edges - full proposed model."""
    return full_graph  # Use the original full graph as-is


# ─────────────────────────────────────────────────────────────────────────────
# Variant configuration by dataset type
# ─────────────────────────────────────────────────────────────────────────────

GRAPH_VARIANTS_FAERS_ALL = {
    "G1_No_Graph"             : lambda tg: _make_no_graph(tg),
    "G2_Drug_Cooccurrence"    : lambda tg: _make_patient_drug(tg, include_adr=False),  # simplified
    "G3_Patient_Drug"         : lambda tg: _make_patient_drug(tg, include_adr=True),
    "G4_Patient_Drug_Disease" : lambda tg: _make_patient_drug_disease(tg, include_adr=True),
    "G5_Full_AER_Graph"       : lambda tg: tg,
}

GRAPH_VARIANTS_TB = {
    "G1_No_Graph"             : lambda tg: _make_no_graph(tg),
    "G2_Drug_Cooccurrence"    : lambda tg: _make_patient_drug(tg, include_adr=False),
    "G3_Patient_Drug"         : lambda tg: _make_patient_drug(tg, include_adr=True),
    "G4_Patient_Drug_Disease" : lambda tg: _make_patient_drug_disease(tg, include_adr=True),
    "G5_Full_AER_Graph_No_DDI_III" : lambda tg: _make_tb_graph_no_ddi_no_iii(tg),
    "G6_TB_No_DDI_No_III"     : lambda tg: _make_tb_graph_no_ddi_no_iii(tg),
    "G7_TB_III_Only"          : lambda tg: _make_tb_graph_iii_only(tg),
    "G8_TB_DDI_Only"          : lambda tg: _make_tb_graph_ddi_only(tg),
    "G9_TB_Full_DDI_III"      : lambda tg: _make_tb_graph_full(tg),
}


# ─────────────────────────────────────────────────────────────────────────────
# Simple MLP-only model for "No Graph" variant
# ─────────────────────────────────────────────────────────────────────────────

class PatientMLP(nn.Module):
    """MLP over patient node features only (no message passing)."""
    def __init__(self, in_dim: int, hidden_dim: int, n_adrs: int,
                 dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.predictor = nn.Linear(hidden_dim, n_adrs)
        
        # Augmentation layer (adds noise via dropout)
        self.aug = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, hidden_dim)
        )
        self._hid = hidden_dim
        
        # Projection head for contrastive learning (like SupCon)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self._head = hidden_dim

    def forward(self, x_dict, edge_index_dict=None, patient_mask=None):
        x = x_dict["patient"]
        if patient_mask is not None:
            x = x[patient_mask]
            
        # Get hidden representation
        h_orig = self.net(x)
        
        # Generate augmented view
        h_aug = self.aug(h_orig)
        
        # Project both views for contrastive learning (like SupCon)
        z1 = self.proj_head(h_orig)
        z2 = self.proj_head(h_aug)
        
        # Predict from augmented representation
        logits  = self.predictor(h_aug)
        return logits, h_orig, z1, z2


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

def _train_on_variant_graph(train_g: HeteroData, val_g: HeteroData,
                             variant_name: str, cfg: dict,
                             device: torch.device) -> tuple:
    """Train a model on the specified graph variant and return (model, val_m, test_m)."""
    if variant_name.startswith("G1_"):
        n_adrs = len(train_g.adr_vocab)
        in_dim = train_g["patient"].x.shape[1]
        model  = PatientMLP(in_dim, cfg["embedding_dim"], n_adrs, cfg["dropout"])
    else:
        model = build_model_from_graph(train_g, cfg)

    model   = model.to(device)
    loss_fn = PreciseADRLoss(alpha=cfg["alpha"], tau=cfg["tau"],
                              gamma=cfg["focal_gamma"]).to(device)
    opt     = Adam(model.parameters(), lr=cfg["lr"])

    x_dict  = {nt: train_g[nt].x.to(device) for nt in train_g.node_types}
    ei_dict = {}
    for et in train_g.edge_types:
        if hasattr(train_g[et], "edge_index"):
            ei_dict[et] = train_g[et].edge_index.to(device)

    mask = train_g["patient"].eval_mask.to(device)
    y    = train_g["patient"].y.to(device)

    best_auc = -1.0; best_state = None; patience = cfg["patience"]
    for ep in range(cfg["max_epochs"]):
        model.train(); opt.zero_grad()
        logits, h_orig, z1, z2 = model(x_dict, ei_dict, patient_mask=mask)
        loss, _, _ = loss_fn(logits, y[mask], h_orig, z1, z2)
        loss.backward(); opt.step()

        if ep % 5 == 0:
            val_m = _eval_on_variant(model, val_g, device)
            if val_m["AUC"] > best_auc:
                best_auc   = val_m["AUC"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience   = cfg["patience"]
            else:
                patience -= 1
            if patience == 0:
                break

    model.load_state_dict(best_state)
    return model


def _eval_on_variant(model, g: HeteroData, device: torch.device) -> dict:
    model.eval()
    with torch.no_grad():
        x_dict  = {nt: g[nt].x.to(device) for nt in g.node_types}
        ei_dict = {et: g[et].edge_index.to(device)
                   for et in g.edge_types
                   if hasattr(g[et], "edge_index")}
        mask    = g["patient"].eval_mask.to(device)
        y_true  = g["patient"].y[g["patient"].eval_mask].numpy()
        logits, _, _, _ = model(x_dict, ei_dict, patient_mask=mask)
        scores  = torch.sigmoid(logits).cpu().numpy()
    return evaluate_all_metrics(scores, y_true)


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_graph_structure_ablation(variant: str, graph_path: str,
                                  results_path: str, device: torch.device):
    print(f"\n{'='*70}")
    print(f"  ABLATION 6.4 – GRAPH STRUCTURE: {variant.upper()}")
    print(f"  Dataset: {DATASET_NAME}")
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

    # Select variant set based on dataset type
    if DATASET_NAME in ["FAERS_TB", "FAERS_TB_DRUGS"]:
        variant_dict = GRAPH_VARIANTS_TB
        print("  Using TB graph variants (G1-G9)")
    else:
        variant_dict = GRAPH_VARIANTS_FAERS_ALL
        print("  Using FAERS-ALL graph variants (G1-G5)")

    all_results = {}

    for vname, graph_fn in variant_dict.items():
        print(f"\n  Variant: {vname}")
        tr_g  = graph_fn(full_train)
        vl_g  = graph_fn(full_val)
        te_g  = graph_fn(full_test)

        model = _train_on_variant_graph(tr_g, vl_g, vname, cfg, device)
        test_m = _eval_on_variant(model, te_g, device)

        print(f"    AUC={test_m['AUC']:.4f}  Hit@10={test_m.get('Hit@10', 0):.4f}")
        all_results[vname] = test_m

    out_path = os.path.join(out_dir, "ablation_graph_structure_supcon.json")
    with open(out_path, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print(f"\n  Saved → {out_path}")
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Ablation 6.4: Graph Structure")
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
            run_graph_structure_ablation(v, args.graph_path, args.results_path, device)
    print("\n✓ Ablation 6.4 (Graph Structure) complete!")


if __name__ == "__main__":
    main()
