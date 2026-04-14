"""
diagnose_training.py
====================
Diagnostic script to identify training issues and poor performance.

Usage:
    python diagnose_training.py --variant xxx
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from config import GRAPH_PATH, MODEL_PATH
from step3_model import build_model_from_graph
from scripts3.step4_training_supcon import evaluate_all_metrics

def analyze_graph_data(graph, split_name):
    """Analyze graph data quality and statistics."""
    print(f"\n{'='*70}")
    print(f"  {split_name.upper()} GRAPH ANALYSIS")
    print(f"{'='*70}")
    
    # Node statistics
    print("\n1. NODE STATISTICS:")
    for node_type in graph.node_types:
        print(f"\n  {node_type}:")
        print(f"    num_nodes: {graph[node_type].num_nodes}")
        if hasattr(graph[node_type], 'x'):
            x = graph[node_type].x
            print(f"    features shape: {x.shape}")
            print(f"    features dtype: {x.dtype}")
            print(f"    features mean: {x.float().mean():.4f}")
            print(f"    features std: {x.float().std():.4f}")
            print(f"    features min/max: {x.min():.4f} / {x.max():.4f}")
            print(f"    features sparsity: {(x == 0).float().mean():.4f}")
    
    # Patient label analysis
    print("\n2. PATIENT LABEL ANALYSIS:")
    y = graph["patient"].y
    mask = graph["patient"].eval_mask
    print(f"  labels shape: {y.shape}")
    print(f"  labels dtype: {y.dtype}")
    print(f"  eval_mask: {mask.sum()} / {mask.shape[0]} patients")
    
    # Check if mask shape matches y shape
    if mask.shape[0] != y.shape[0]:
        print(f"\n  ⚠️ WARNING: Mask shape mismatch!")
        print(f"    mask length: {mask.shape[0]}")
        print(f"    labels rows: {y.shape[0]}")
        print(f"    This is a DATA BUG - mask should match label rows!")
        # Use only the patients that have labels
        y_masked = y
    else:
        # Check if labels are in mask
        y_masked = y[mask]
    
    print(f"\n  Masked patients:")
    print(f"    shape: {y_masked.shape}")
    print(f"    positive labels per patient: {y_masked.sum(dim=1).float().mean():.2f}")
    print(f"    total positive: {y_masked.sum():.0f}")
    print(f"    label density: {y_masked.float().mean():.6f}")
    print(f"    max labels per patient: {y_masked.sum(dim=1).max()}")
    print(f"    min labels per patient: {y_masked.sum(dim=1).min()}")
    
    # Check label distribution
    label_counts = y_masked.sum(dim=0)
    print(f"\n  Label distribution across ADRs:")
    print(f"    ADRs with 0 positives: {(label_counts == 0).sum()}")
    print(f"    ADRs with 1-10 positives: {((label_counts > 0) & (label_counts <= 10)).sum()}")
    print(f"    ADRs with 10-100 positives: {((label_counts > 10) & (label_counts <= 100)).sum()}")
    print(f"    ADRs with >100 positives: {(label_counts > 100).sum()}")
    print(f"    Most common ADR count: {label_counts.max()}")
    print(f"    Least common ADR count (>0): {label_counts[label_counts > 0].min()}")
    
    # Edge statistics
    print("\n3. EDGE STATISTICS:")
    for edge_type in graph.edge_types:
        if hasattr(graph[edge_type], 'edge_index'):
            edge_index = graph[edge_type].edge_index
            print(f"  {edge_type}: {edge_index.shape[1]:,} edges")
            
            # Check for self-loops
            if edge_type[0] == edge_type[2]:
                src, dst = edge_index
                self_loops = (src == dst).sum()
                print(f"    self-loops: {self_loops}")


def test_model_forward(model, graph, device):
    """Test if model forward pass works correctly."""
    print(f"\n{'='*70}")
    print("  MODEL FORWARD PASS TEST")
    print(f"{'='*70}")
    
    model = model.to(device)
    model.eval()
    
    x_dict = {nt: graph[nt].x.to(device) for nt in graph.node_types}
    edge_idx_dict = {et: graph[et].edge_index.to(device)
                     for et in graph.edge_types
                     if hasattr(graph[et], "edge_index")}
    mask = graph["patient"].eval_mask.to(device)
    
    print(f"\n1. Input shapes:")
    for nt, x in x_dict.items():
        print(f"  {nt}: {x.shape}")
    
    with torch.no_grad():
        logits, h_orig, h_aug = model(x_dict, edge_idx_dict, patient_mask=mask)
    
    print(f"\n2. Output shapes:")
    print(f"  logits: {logits.shape}")
    print(f"  h_orig: {h_orig.shape}")
    print(f"  h_aug: {h_aug.shape}")
    
    print(f"\n3. Output statistics:")
    print(f"  logits mean: {logits.mean():.4f}")
    print(f"  logits std: {logits.std():.4f}")
    print(f"  logits min/max: {logits.min():.4f} / {logits.max():.4f}")
    
    probs = torch.sigmoid(logits)
    print(f"  probs mean: {probs.mean():.4f}")
    print(f"  probs std: {probs.std():.4f}")
    print(f"  probs min/max: {probs.min():.4f} / {probs.max():.4f}")
    
    # Check if model is predicting all same
    unique_predictions = torch.unique(probs.round(decimals=2)).numel()
    print(f"  unique probability values (rounded): {unique_predictions}")
    
    if unique_predictions < 10:
        print("  ⚠️ WARNING: Model outputs very few unique values (might be stuck)")
    
    return logits, h_orig, h_aug


def analyze_loss_components(model, graph, device, alpha=0.5, tau=0.05, gamma=2.0):
    """Analyze individual loss components."""
    print(f"\n{'='*70}")
    print("  LOSS COMPONENT ANALYSIS")
    print(f"{'='*70}")
    
    from step3_model import PreciseADRLoss
    
    model = model.to(device)
    model.eval()
    loss_fn = PreciseADRLoss(alpha=alpha, tau=tau, gamma=gamma).to(device)
    
    x_dict = {nt: graph[nt].x.to(device) for nt in graph.node_types}
    edge_idx_dict = {et: graph[et].edge_index.to(device)
                     for et in graph.edge_types
                     if hasattr(graph[et], "edge_index")}
    mask = graph["patient"].eval_mask.to(device)
    y_all = graph["patient"].y.to(device)
    
    with torch.no_grad():
        logits, h_orig, h_aug = model(x_dict, edge_idx_dict, patient_mask=mask)
        # Extract labels only for eval patients (matching logits shape)
        y = y_all[mask]
        total, l_focal, l_nce = loss_fn(logits, y, h_orig, h_aug)
    
    print(f"\n  Loss values:")
    print(f"    Focal loss: {l_focal.item():.4f}")
    print(f"    InfoNCE loss: {l_nce.item():.4f}")
    print(f"    Total loss: {total.item():.4f}")
    print(f"    Expected total: {alpha * l_nce.item() + (1-alpha) * l_focal.item():.4f}")
    
    print(f"\n  Loss ratios:")
    print(f"    InfoNCE / Focal: {l_nce.item() / (l_focal.item() + 1e-8):.2f}x")
    print(f"    Weighted InfoNCE: {alpha * l_nce.item():.4f}")
    print(f"    Weighted Focal: {(1-alpha) * l_focal.item():.4f}")
    
    # Check if InfoNCE is dominating
    if l_nce.item() / (l_focal.item() + 1e-8) > 10:
        print("\n  ⚠️ WARNING: InfoNCE loss is much larger than Focal loss!")
        print(f"  → Consider decreasing α (currently {alpha}) or increasing τ (currently {tau})")
    
    # Analyze embedding similarity
    print(f"\n  Embedding analysis:")
    h_orig_norm = F.normalize(h_orig, dim=-1)
    h_aug_norm = F.normalize(h_aug, dim=-1)
    
    # Positive pairs (same patient)
    pos_sim = (h_orig_norm * h_aug_norm).sum(dim=-1)
    print(f"    Positive pair similarity: {pos_sim.mean():.4f} ± {pos_sim.std():.4f}")
    
    # Negative pairs (different patients) - sample for efficiency
    n_samples = min(1000, h_orig.shape[0])
    indices = torch.randperm(h_orig.shape[0])[:n_samples]
    h1_sample = h_orig_norm[indices]
    h2_sample = h_aug_norm[torch.roll(indices, 1)]  # Shifted indices
    neg_sim = (h1_sample * h2_sample).sum(dim=-1)
    print(f"    Negative pair similarity: {neg_sim.mean():.4f} ± {neg_sim.std():.4f}")
    
    if pos_sim.mean() < neg_sim.mean() + 0.1:
        print("\n  ⚠️ WARNING: Positive and negative pairs have similar similarity!")
        print("  → Contrastive learning is not working properly")


def check_random_baseline(graph):
    """Check performance of random predictions."""
    print(f"\n{'='*70}")
    print("  RANDOM BASELINE")
    print(f"{'='*70}")
    
    mask = graph["patient"].eval_mask
    y = graph["patient"].y[mask].numpy()
    
    # Random predictions
    n_patients, n_adrs = y.shape
    random_scores = np.random.rand(n_patients, n_adrs)
    
    metrics = evaluate_all_metrics(random_scores, y)
    print(f"\n  Random prediction metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="xxx")
    parser.add_argument("--graph-path", default=GRAPH_PATH)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    var_graph_dir = os.path.join(args.graph_path, args.variant)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING DIAGNOSTICS: {args.variant.upper()}")
    print(f"{'='*70}")
    
    # Load graphs
    print(f"\nLoading graphs from {var_graph_dir}...")
    train_graph = torch.load(os.path.join(var_graph_dir, "train_graph.pt"), weights_only=False)
    val_graph = torch.load(os.path.join(var_graph_dir, "val_graph.pt"), weights_only=False)
    
    # Analyze graph data
    analyze_graph_data(train_graph, "train")
    analyze_graph_data(val_graph, "validation")
    
    # Build and test model
    print(f"\nBuilding model...")
    from config import MODEL as MODEL_CFG
    model = build_model_from_graph(train_graph, MODEL_CFG)
    
    # Test model forward pass
    test_model_forward(model, train_graph, device)
    
    # Analyze loss components
    analyze_loss_components(model, train_graph, device,
                           alpha=MODEL_CFG["alpha"],
                           tau=MODEL_CFG["tau"],
                           gamma=MODEL_CFG["focal_gamma"])
    
    # Check random baseline
    check_random_baseline(val_graph)
    
    # Load trained model if exists
    ckpt_path = os.path.join(args.model_path, f"{args.variant}_best.pt")
    if os.path.exists(ckpt_path):
        print(f"\n{'='*70}")
        print("  TRAINED MODEL ANALYSIS")
        print(f"{'='*70}")
        
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"\n  Checkpoint info:")
        print(f"    α: {checkpoint['alpha']}")
        print(f"    τ: {checkpoint['tau']}")
        print(f"    Val AUC: {checkpoint['val_metrics']['AUC']:.4f}")
        print(f"    Test AUC: {checkpoint['test_metrics']['AUC']:.4f}")
        
        # Test trained model
        test_model_forward(model, val_graph, device)
        analyze_loss_components(model, val_graph, device,
                               alpha=checkpoint['alpha'],
                               tau=checkpoint['tau'],
                               gamma=MODEL_CFG["focal_gamma"])
    
    print(f"\n{'='*70}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
