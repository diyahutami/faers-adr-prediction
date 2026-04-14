"""
step4_training.py
=================
Step 4 of the PreciseADR reproduction pipeline.

Trains PreciseADR on each dataset variant (XXX, XXX-Gender, XXX-Age) with:
  • HGT backbone
  • Patient Node Augmentation + InfoNCE contrastive loss
  • Focal Loss for imbalanced multi-label classification
  • Grid search over α (InfoNCE weight) × τ (temperature)
  • Early stopping on validation AUC

Evaluation metrics reported:
  AUC (macro), Hit@K (K=1,2,5,10,20), NDCG@K, Recall@K

Usage
-----
    # Train all variants with grid search:
    python step4_training.py

    # Train one variant with fixed hyperparameters:
    python step4_training.py --variant xxx --alpha 0.5 --tau 0.05 --no-grid-search
    
    # Train with reproducible results (using seed):
    python step4_training.py --seed 42
"""

import os
import sys
import json
import time
import argparse
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config_2018 import DATA_PATH, OUTPUT_PATH, GRAPH_PATH, MODEL_PATH, RESULTS_PATH, MODEL, EVAL
from config_2018 import DATASET_NAME, OUTPUT_PATH
from step3_model_infonce import PreciseADR, PreciseADRLoss, build_model_from_graph

METHOD = "infonce"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_infonce")

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

# ─────────────────────────────────────────────────────────────────────────────
# Helper function for JSON serialization
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_serializable(obj):
    """Convert numpy/torch types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    else:
        return obj

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged AUC across all ADR classes with >0 positive samples."""
    valid_cols = labels.sum(axis=0) > 0
    if valid_cols.sum() == 0:
        return 0.0
    return roc_auc_score(labels[:, valid_cols], scores[:, valid_cols],
                         average="macro")


def compute_hit_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Fraction of patients with at least one true ADR in top-K predicted."""
    top_k_idx = np.argsort(-scores, axis=1)[:, :k]
    hits = 0
    for i, row in enumerate(labels):
        if row[top_k_idx[i]].sum() > 0:
            hits += 1
    return hits / len(labels)


def compute_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    """NDCG@K averaged over patients."""
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
    """Mean Recall@K across patients."""
    recalls = []
    for i in range(len(labels)):
        top_k = np.argsort(-scores[i])[:k]
        n_pos = labels[i].sum()
        if n_pos == 0:
            continue
        recalls.append(labels[i][top_k].sum() / n_pos)
    return np.mean(recalls) if recalls else 0.0


def evaluate_all_metrics(scores: np.ndarray, labels: np.ndarray,
                         k_values: list = None) -> dict:
    if k_values is None:
        k_values = EVAL["hit_k"]
    metrics = {"AUC": compute_auc(scores, labels)}
    for k in k_values:
        metrics[f"Hit@{k}"]   = compute_hit_at_k(scores, labels, k)
        metrics[f"NDCG@{k}"]  = compute_ndcg_at_k(scores, labels, k)
        metrics[f"Recall@{k}"] = compute_recall_at_k(scores, labels, k)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_one_epoch(model: PreciseADR, graph, loss_fn: PreciseADRLoss,
                  optimizer, device: torch.device,
                  is_train: bool = True, scaler=None, 
                  accumulation_steps: int = 1) -> dict:
    """Run one forward pass over the graph (full-batch) with gradient accumulation."""
    model.train(is_train)

    x_dict        = {nt: graph[nt].x.to(device) for nt in graph.node_types}
    edge_idx_dict = {et: graph[et].edge_index.to(device)
                     for et in graph.edge_types
                     if hasattr(graph[et], "edge_index")}
    mask  = graph["patient"].eval_mask.to(device)
    y_all = graph["patient"].y.to(device)
    # Extract labels only for eval patients (matching model output)
    y = y_all[mask]

    if is_train:
        optimizer.zero_grad()

    # Use automatic mixed precision if scaler is provided
    if scaler is not None and is_train:
        with autocast(device_type=device.type, dtype=torch.float16):
            logits, h_orig, h_aug = model(x_dict, edge_idx_dict, patient_mask=mask)
            total, l_focal, l_nce = loss_fn(logits, y, h_orig, h_aug)
            total = total / accumulation_steps
        
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits, h_orig, h_aug = model(x_dict, edge_idx_dict, patient_mask=mask)
        total, l_focal, l_nce = loss_fn(logits, y, h_orig, h_aug)
        
        if is_train:
            (total / accumulation_steps).backward()
            optimizer.step()

    # Clear cache to free memory
    if is_train and device.type == 'cuda':
        torch.cuda.empty_cache()

    return {
        "loss"   : total.item() * accumulation_steps,
        "focal"  : l_focal.item(),
        "infonce": l_nce.item(),
    }


def evaluate(model: PreciseADR, graph, device: torch.device) -> dict:
    """Compute AUC and Hit@K on a graph split."""
    model.eval()
    with torch.no_grad():
        x_dict        = {nt: graph[nt].x.to(device) for nt in graph.node_types}
        edge_idx_dict = {et: graph[et].edge_index.to(device)
                         for et in graph.edge_types
                         if hasattr(graph[et], "edge_index")}
        mask = graph["patient"].eval_mask.to(device)
        y_all= graph["patient"].y
        # Extract labels only for eval patients (matching model output)
        y     = y_all[mask.cpu()].numpy()
        logits, _, _ = model(x_dict, edge_idx_dict, patient_mask=mask)
        scores = torch.sigmoid(logits).cpu().numpy()
    return evaluate_all_metrics(scores, y)


def train_model(train_graph, val_graph, model: PreciseADR,
                alpha: float, tau: float, gamma: float,
                lr: float, max_epochs: int, patience: int,
                device: torch.device, use_amp: bool = True,
                accumulation_steps: int = 1) -> tuple[PreciseADR, dict, dict]:
    """
    Train PreciseADR and return (best_model, best_val_metrics, history).
    
    Args:
        use_amp: Use automatic mixed precision (FP16) training
        accumulation_steps: Number of gradient accumulation steps
    """
    # Reset seed for reproducibility if seed was set
    if _GLOBAL_SEED is not None:
        set_seed(_GLOBAL_SEED, verbose=False)
    
    model = model.to(device)
    loss_fn   = PreciseADRLoss(alpha=alpha, tau=tau, gamma=gamma).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler() if (use_amp and device.type == 'cuda') else None

    best_auc      = -1.0
    best_state    = None
    patience_left = patience

    history = {"train_loss": [], "val_auc": []}

    for epoch in range(1, max_epochs + 1):
        train_info = run_one_epoch(model, train_graph, loss_fn, optimizer,
                                   device, is_train=True, scaler=scaler,
                                   accumulation_steps=accumulation_steps)
        val_metrics = evaluate(model, val_graph, device)
        val_auc     = val_metrics["AUC"]

        history["train_loss"].append(train_info["loss"])
        history["val_auc"].append(val_auc)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if epoch % 10 == 0 or patience_left == 0:
            print(f"  Epoch {epoch:3d} | loss={train_info['loss']:.4f} "
                  f"focal={train_info['focal']:.4f} nce={train_info['infonce']:.4f} "
                  f"val_AUC={val_auc:.4f} | best={best_auc:.4f}")

        if patience_left == 0:
            print(f"  Early stopping at epoch {epoch}")
            break
        
        # Clear cache after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    model.load_state_dict(best_state)
    best_val_metrics = evaluate(model, val_graph, device)
    return model, best_val_metrics, history


# ─────────────────────────────────────────────────────────────────────────────
# Grid search over α × τ
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(train_graph, val_graph, base_model_cfg: dict,
                alpha_range: list, tau_range: list,
                device: torch.device, out_dir: str) -> tuple[float, float]:
    """
    Perform α × τ grid search on validation AUC.
    Returns (best_alpha, best_tau).
    Saves a 2-D results table as grid_search_results.json.
    """
    print(f"\n  Grid search: {len(alpha_range)} α × {len(tau_range)} τ "
          f"= {len(alpha_range)*len(tau_range)} runs")
    grid_results = {}
    best_auc = -1.0
    best_alpha, best_tau = alpha_range[0], tau_range[0]

    for alpha, tau in itertools.product(alpha_range, tau_range):
        # Reset seed for reproducibility if seed was set
        if _GLOBAL_SEED is not None:
            set_seed(_GLOBAL_SEED, verbose=False)
        
        # Re-instantiate fresh model for each grid point
        model = build_model_from_graph(train_graph, base_model_cfg)
        trained, val_metrics, _ = train_model(
            train_graph, val_graph, model,
            alpha=alpha, tau=tau,
            gamma=base_model_cfg["focal_gamma"],
            lr=base_model_cfg["lr"],
            max_epochs=50,      # shorter run during grid search
            patience=10,
            device=device,
            use_amp=base_model_cfg.get("use_amp", True),
            accumulation_steps=base_model_cfg.get("accumulation_steps", 1),
        )
        auc = val_metrics["AUC"]
        grid_results[f"a{alpha}_t{tau}"] = {
            "alpha": float(alpha), 
            "tau": float(tau), 
            "val_auc": float(auc)
        }
        print(f"  α={alpha} τ={tau} → val_AUC={auc:.4f}")
        if auc > best_auc:
            best_auc   = auc
            best_alpha = alpha
            best_tau   = tau

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "grid_search_results.json"), "w") as f:
        json.dump(convert_to_serializable(grid_results), f, indent=2)
    print(f"\n  Best: α={best_alpha}  τ={best_tau}  val_AUC={best_auc:.4f}")
    return best_alpha, best_tau


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine per variant
# ─────────────────────────────────────────────────────────────────────────────

def train_variant(variant: str, cfg: dict, graph_path: str,
                  model_path: str, results_path: str,
                  do_grid_search: bool, device: torch.device):
    print(f"\n{'='*70}")
    print(f"  TRAINING: {variant.upper()}")
    print(f"{'='*70}")

    var_graph_dir = os.path.join(graph_path, variant)
    out_dir       = os.path.join(results_path, variant)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    # Clear GPU cache before loading graphs
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"  GPU memory before loading: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

    # Load graphs (weights_only=False needed for PyTorch Geometric objects)
    train_graph = torch.load(os.path.join(var_graph_dir, "train_graph.pt"), weights_only=False)
    val_graph   = torch.load(os.path.join(var_graph_dir, "val_graph.pt"), weights_only=False)
    test_graph  = torch.load(os.path.join(var_graph_dir, "test_graph.pt"), weights_only=False)
    
    if device.type == 'cuda':
        print(f"  GPU memory after loading: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")

    # Grid search
    if do_grid_search:
        best_alpha, best_tau = grid_search(
            train_graph, val_graph,
            base_model_cfg=cfg,
            alpha_range=cfg["alpha_range"],
            tau_range=cfg["tau_range"],
            device=device,
            out_dir=out_dir,
        )
    else:
        best_alpha = cfg["alpha"]
        best_tau   = cfg["tau"]

    # Full training with best hyperparameters
    print(f"\n  Full training: α={best_alpha}  τ={best_tau}")
    
    # Reset seed for reproducibility if seed was set
    if _GLOBAL_SEED is not None:
        set_seed(_GLOBAL_SEED, verbose=False)
    
    model = build_model_from_graph(train_graph, cfg)
    model, val_metrics, history = train_model(
        train_graph, val_graph, model,
        alpha=best_alpha, tau=best_tau,
        gamma=cfg["focal_gamma"],
        lr=cfg["lr"],
        max_epochs=cfg["max_epochs"],
        patience=cfg["patience"],
        device=device,
        use_amp=cfg.get("use_amp", True),
        accumulation_steps=cfg.get("accumulation_steps", 1),
    )

    # Test evaluation
    test_metrics = evaluate(model, test_graph, device)

    print(f"\n  Validation metrics:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save model checkpoint
    ckpt_path = os.path.join(model_path, f"{variant}_best_infonce.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "alpha"           : best_alpha,
        "tau"             : best_tau,
        "val_metrics"     : val_metrics,
        "test_metrics"    : test_metrics,
        "history"         : history,
    }, ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")

    # Save metrics JSON
    metrics_path = os.path.join(out_dir, "final_metrics_infonce.json")
    metrics_data = {
        "variant": variant, 
        "alpha": float(best_alpha), 
        "tau": float(best_tau),
        "val_metrics": convert_to_serializable(val_metrics), 
        "test_metrics": convert_to_serializable(test_metrics)
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)

    return model, test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PreciseADR Step 4: Training")
    p.add_argument("--graph-path",     default=GRAPH_PATH)
    p.add_argument("--model-path",     default=MODEL_PATH)
    p.add_argument("--results-path",   default=RESULTS_PATH)
    p.add_argument("--variant",        default="all",
                   choices=["all", "xxx", "xxx_gender", "xxx_age"])
    p.add_argument("--alpha",          type=float, default=None)
    p.add_argument("--tau",            type=float, default=None)
    p.add_argument("--no-grid-search", action="store_true")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int, default=123,
                   help="Random seed for reproducibility (default: None, no seed set)")
    return p.parse_args()


def main():
    args   = parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        set_seed(args.seed)
    
    device = torch.device(args.device)
    cfg    = dict(MODEL)   # copy from config

    if args.alpha is not None: cfg["alpha"] = args.alpha
    if args.tau   is not None: cfg["tau"]   = args.tau

    do_grid = not args.no_grid_search

    variants = (["xxx", "xxx_gender", "xxx_age"]
                if args.variant == "all" else [args.variant])

    all_results = {}
    for variant in variants:
        graph_dir = os.path.join(args.graph_path, variant)
        if not os.path.exists(graph_dir):
            print(f"  ⚠ Graph not found for {variant}, skipping.")
            continue
        _, metrics = train_variant(
            variant, cfg,
            args.graph_path, args.model_path, args.results_path,
            do_grid_search=do_grid, device=device,
        )
        all_results[variant] = metrics

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for v, m in all_results.items():
        print(f"  {v:15s}  AUC={m['AUC']:.4f}  Hit@10={m.get('Hit@10', 0):.4f}")

    os.makedirs(args.results_path, exist_ok=True)
    with open(os.path.join(args.results_path, "training_summary_infonce.json"), "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print("\n✓ Step 4 (Training) complete!")


if __name__ == "__main__":
    main()
