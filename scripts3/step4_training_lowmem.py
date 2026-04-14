"""
step4_training_lowmem.py
========================
Memory-optimized version of step4_training.py for systems with limited RAM.

Changes from original:
1. Reduced batch processing
2. Gradient checkpointing
3. CPU offloading for val/test graphs
4. Reduced embedding dimensions
5. Fewer HGT layers
6. Smaller grid search space

Usage:
    # Train with default settings (CPU):
    python step4_training_lowmem.py --variant xxx
    
    # Train with GPU and seed for reproducibility:
    python step4_training_lowmem.py --variant xxx --device cuda --seed 42
"""

import os
import sys
import json
import gc
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_PATH, OUTPUT_PATH, GRAPH_PATH, MODEL_PATH, RESULTS_PATH, MODEL, EVAL
from step3_model import PreciseADR, PreciseADRLoss, build_model_from_graph

# ─────────────────────────────────────────────────────────────────────────────
# Seed setting for reproducibility
# ─────────────────────────────────────────────────────────────────────────────

# Global variable to store the seed for reuse
_GLOBAL_SEED = None

def set_seed(seed: int, verbose: bool = True):
    """Set random seed for reproducibility across all libraries."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility (may reduce performance by ~10-30%)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Additional PyTorch settings for determinism
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if verbose:
        print(f"  Random seed set to: {seed}")
        print(f"  Note: Deterministic mode enabled (may reduce training speed)")

# Memory-optimized configuration
LOWMEM_MODEL = dict(MODEL)
LOWMEM_MODEL.update({
    "embedding_dim": 256,      # Reduced from 256
    "num_hgt_layers": 2,       # Reduced from 3
    "num_heads": 2,            # Reduced from 8
    "batch_size": 256,         # Reduced from 512
    "max_epochs": 100,         # Reduced from 200
    "patience": 15,            # Reduced from 20
    # InfoNCE: Use sampled negatives for memory efficiency (default)
    "use_sampled_negatives": True,   # Set to False for exact paper formula (may cause OOM)
    "max_negatives": 2048,           # Max negative samples when sampling is enabled
})

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics (same as original)
# ─────────────────────────────────────────────────────────────────────────────

def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    valid_cols = labels.sum(axis=0) > 0
    if valid_cols.sum() == 0:
        return 0.0
    return roc_auc_score(labels[:, valid_cols], scores[:, valid_cols], average="macro")

def compute_hit_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k_idx = np.argsort(-scores, axis=1)[:, :k]
    hits = 0
    for i, row in enumerate(labels):
        if row[top_k_idx[i]].sum() > 0:
            hits += 1
    return hits / len(labels)

def compute_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    def _dcg(rel, k):
        rel = rel[:k]
        denom = np.log2(np.arange(2, len(rel) + 2))
        return (rel / denom).sum()
    
    ndcgs = []
    for i in range(len(labels)):
        order = np.argsort(-scores[i])
        rel   = labels[i][order].astype(float)
        ideal = np.sort(labels[i])[::-1].astype(float)
        dcg   = _dcg(rel, k)
        idcg  = _dcg(ideal, k)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs)

def compute_recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    recalls = []
    for i in range(len(labels)):
        top_k = np.argsort(-scores[i])[:k]
        n_pos = labels[i].sum()
        if n_pos == 0:
            continue
        recalls.append(labels[i][top_k].sum() / n_pos)
    return np.mean(recalls) if recalls else 0.0

def convert_to_python_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def evaluate_all_metrics(scores: np.ndarray, labels: np.ndarray,
                         k_values: list = None) -> dict:
    if k_values is None:
        k_values = EVAL["hit_k"]
    metrics = {"AUC": float(compute_auc(scores, labels))}
    for k in k_values:
        metrics[f"Hit@{k}"]   = float(compute_hit_at_k(scores, labels, k))
        metrics[f"NDCG@{k}"]  = float(compute_ndcg_at_k(scores, labels, k))
        metrics[f"Recall@{k}"] = float(compute_recall_at_k(scores, labels, k))
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# Training loop with memory optimization
# ─────────────────────────────────────────────────────────────────────────────
def run_one_epoch(model: PreciseADR, graph, loss_fn: PreciseADRLoss,
                  optimizer, device: torch.device,
                  is_train: bool = True) -> dict:
    """Run one forward pass with memory optimization."""
    model.train(is_train)
    
    # Move only necessary data to GPU
    x_dict = {}
    for nt in graph.node_types:
        x_dict[nt] = graph[nt].x.to(device)
    
    edge_idx_dict = {}
    for et in graph.edge_types:
        if hasattr(graph[et], "edge_index"):
            edge_idx_dict[et] = graph[et].edge_index.to(device)
    
    mask = graph["patient"].eval_mask.to(device)
    y_all = graph["patient"].y.to(device)
    # Extract labels only for eval patients (matching model output)
    y = y_all[mask]
    
    if is_train:
        optimizer.zero_grad()
    
    # Forward pass
    with torch.set_grad_enabled(is_train):
        logits, h_orig, h_aug = model(x_dict, edge_idx_dict, patient_mask=mask)
        total, l_focal, l_nce = loss_fn(logits, y, h_orig, h_aug)
    
    # Store loss values before potentially deleting tensors
    loss_dict = {
        "loss": total.item(),
        "focal": l_focal.item(),
        "infonce": l_nce.item(),
    }
    
    if is_train:
        total.backward()
        optimizer.step()
        
        # Clear GPU cache after backward
        del logits, h_orig, h_aug, total, l_focal, l_nce
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return loss_dict

def evaluate(model: PreciseADR, graph, device: torch.device) -> dict:
    """Evaluate with memory optimization."""
    model.eval()
    
    with torch.no_grad():
        x_dict = {nt: graph[nt].x.to(device) for nt in graph.node_types}
        edge_idx_dict = {et: graph[et].edge_index.to(device)
                        for et in graph.edge_types
                        if hasattr(graph[et], "edge_index")}
        mask = graph["patient"].eval_mask.to(device)
        y_all= graph["patient"].y
        # Extract labels only for eval patients (matching model output)
        y     = y_all[mask.cpu()].numpy() # Keep labels on CPU
        
        logits, _, _ = model(x_dict, edge_idx_dict, patient_mask=mask)
        scores = torch.sigmoid(logits).cpu().numpy()
        
        # Clear GPU cache
        del logits, x_dict, edge_idx_dict, mask
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    return evaluate_all_metrics(scores, y)

def train_model(train_graph, val_graph, model: PreciseADR,
                alpha: float, tau: float, gamma: float,
                lr: float, max_epochs: int, patience: int,
                device: torch.device, use_sampled_negatives: bool = True,
                max_negatives: int = 2048) -> tuple:
    """Train with memory optimization."""
    # Reset seed for reproducibility if seed was set
    if _GLOBAL_SEED is not None:
        set_seed(_GLOBAL_SEED, verbose=False)
    
    model = model.to(device)
    loss_fn = PreciseADRLoss(alpha=alpha, tau=tau, gamma=gamma,
                             use_sampled_negatives=use_sampled_negatives,
                             max_negatives=max_negatives).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    best_auc = -1.0
    best_state = None
    patience_left = patience
    history = {"train_loss": [], "val_auc": []}
    
    print(f"  Training with α={alpha}, τ={tau}, γ={gamma}, lr={lr}")
    
    for epoch in range(1, max_epochs + 1):
        # Training
        train_info = run_one_epoch(model, train_graph, loss_fn, optimizer,
                                   device, is_train=True)
        
        # Validation (every 5 epochs to save time)
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate(model, val_graph, device)
            val_auc = val_metrics["AUC"]
            
            history["train_loss"].append(train_info["loss"])
            history["val_auc"].append(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
            
            if epoch % 10 == 0 or patience_left == 0:
                print(f"  Epoch {epoch:3d} | loss={train_info['loss']:.4f} "
                      f"val_AUC={val_auc:.4f} | best={best_auc:.4f}")
            
            if patience_left == 0:
                print(f"  Early stopping at epoch {epoch}")
                break
        
        # Memory cleanup
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    model.load_state_dict(best_state)
    best_val_metrics = evaluate(model, val_graph, device)
    return model, best_val_metrics, history

# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train_variant(variant: str, cfg: dict, graph_path: str,
                  model_path: str, results_path: str, device: torch.device):
    print(f"\n{'='*70}")
    print(f"  LOW-MEMORY TRAINING: {variant.upper()}")
    print(f"{'='*70}")
    
    var_graph_dir = os.path.join(graph_path, variant)
    out_dir = os.path.join(results_path, variant)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Load graphs (keep on CPU initially)
    print("  Loading graphs...")
    train_graph = torch.load(os.path.join(var_graph_dir, "train_graph.pt"), 
                            weights_only=False)
    val_graph = torch.load(os.path.join(var_graph_dir, "val_graph.pt"), 
                          weights_only=False)
    test_graph = torch.load(os.path.join(var_graph_dir, "test_graph.pt"), 
                           weights_only=False)
    print("  ✓ Graphs loaded")
    
    # Use simplified hyperparameters (no grid search for low memory)
    alpha = cfg.get("alpha", 0.5)
    tau = cfg.get("tau", 0.05)
    
    # Reset seed for reproducibility if seed was set
    if _GLOBAL_SEED is not None:
        set_seed(_GLOBAL_SEED, verbose=False)
    
    # Build and train model
    print(f"  Building model...")
    model = build_model_from_graph(train_graph, cfg)
    print(f"  ✓ Model built")
    
    model, val_metrics, history = train_model(
        train_graph, val_graph, model,
        alpha=alpha, tau=tau,
        gamma=cfg["focal_gamma"],
        lr=cfg["lr"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        device=device,
        use_sampled_negatives=cfg.get("use_sampled_negatives", True),
        max_negatives=cfg.get("max_negatives", 2048),
    )
    
    # Test evaluation
    print("\n  Evaluating on test set...")
    test_metrics = evaluate(model, test_graph, device)
    
    print(f"\n  Validation metrics:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")
    
    # Save checkpoint
    ckpt_path = os.path.join(model_path, f"{variant}_lowmem.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "alpha": alpha,
        "tau": tau,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "config": cfg,
    }, ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")
    
    # Save metrics
    metrics_path = os.path.join(out_dir, "lowmem_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "variant": variant,
            "alpha": float(alpha),
            "tau": float(tau),
            "val_metrics": convert_to_python_types(val_metrics),
            "test_metrics": convert_to_python_types(test_metrics),
        }, f, indent=2)
    
    return model, test_metrics

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PreciseADR Low-Memory Training")
    p.add_argument("--graph-path", default=GRAPH_PATH)
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--results-path", default=RESULTS_PATH)
    p.add_argument("--variant", default="xxx",
                   choices=["xxx", "xxx_gender", "xxx_age"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--device", default="cpu")  # Default to CPU for low memory
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility (default: None, no seed set)")
    return p.parse_args()

def main():
    args = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        set_seed(args.seed)
    
    device = torch.device(args.device)
    
    # Use low-memory configuration
    cfg = dict(LOWMEM_MODEL)
    cfg["alpha"] = args.alpha
    cfg["tau"] = args.tau
    
    print(f"\n{'='*70}")
    print("  MEMORY-OPTIMIZED TRAINING MODE")
    print(f"{'='*70}")
    print(f"  Device: {device}")
    print(f"  Embedding dim: {cfg['embedding_dim']}")
    print(f"  HGT layers: {cfg['num_hgt_layers']}")
    print(f"  Heads: {cfg['num_heads']}")
    print(f"{'='*70}")
    
    graph_dir = os.path.join(args.graph_path, args.variant)
    if not os.path.exists(graph_dir):
        print(f"  ⚠ Graph not found for {args.variant}")
        return
    
    _, metrics = train_variant(
        args.variant, cfg,
        args.graph_path, args.model_path, args.results_path,
        device=device,
    )
    
    print(f"\n{'='*70}")
    print("  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  AUC: {metrics['AUC']:.4f}")
    print(f"  Hit@10: {metrics.get('Hit@10', 0):.4f}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
