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
  AUC (macro), AUPRC (macro), Hit@K (K=1,2,5,10,20), NDCG@K, Recall@K
  Brier score, ECE (Expected Calibration Error), MCC, Macro-F1, Micro-F1
  Temperature scaling (post-hoc calibration) applied to test evaluation

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
import math
import time
import argparse
import itertools
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, f1_score, matthews_corrcoef,
)

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_PATH, OUTPUT_PATH, GRAPH_PATH, MODEL_PATH, RESULTS_PATH, MODEL, EVAL
from config import DATASET_NAME, OUTPUT_PATH
import config as _cfg_module   # used to patch KG_EDGES at runtime for ablation
from step3_model_infonce import PreciseADR, PreciseADRLoss, build_model_from_graph

METHOD = "infonce"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_infonce")

# ─────────────────────────────────────────────────────────────────────────────
# Seed setting for reproducibility
# ─────────────────────────────────────────────────────────────────────────────

# Global variable to store the seed for reuse
_GLOBAL_SEED = None

def set_seed(seed: int, verbose: bool = True, deterministic: bool = False):
    """Set random seed for reproducibility across all libraries.

    deterministic : if True, force cuDNN deterministic kernels and
                    torch.use_deterministic_algorithms(True).  This guarantees
                    bit-exact reproducibility but serialises CuBLAS operations,
                    causing 10-30% slowdown on CUDA ≥ 10.2.
                    Default False → fast non-deterministic kernels (still seeded,
                    so runs are reproducible within a tolerance of ±1e-5 on GPU).
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # benchmark=True: cuDNN auto-tunes kernels for fixed-size ops → faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if verbose:
        mode = "deterministic (slow)" if deterministic else "non-deterministic (fast)"
        print(f"  Random seed set to: {seed}  [{mode}]")

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


def compute_auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged AUPRC across all ADR classes with >0 positive samples."""
    valid_cols = labels.sum(axis=0) > 0
    if valid_cols.sum() == 0:
        return 0.0
    return float(average_precision_score(
        labels[:, valid_cols], scores[:, valid_cols], average="macro"
    ))


def compute_brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mean Brier score across all ADR classes with >0 positive samples."""
    brier_list = [
        brier_score_loss(labels[:, c], scores[:, c])
        for c in range(labels.shape[1]) if labels[:, c].sum() > 0
    ]
    return float(np.mean(brier_list)) if brier_list else 0.0


def compute_ece(scores: np.ndarray, labels: np.ndarray,
                n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) averaged across ADR classes.
    Bins predicted probabilities into n_bins equal-width buckets and
    measures the mean absolute gap between confidence and accuracy.
    """
    ece_per_class = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        s = scores[:, c]
        l = labels[:, c]
        bin_idx = np.clip(np.digitize(s, bin_edges) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            conf = s[mask].mean()
            acc  = l[mask].mean()
            ece += mask.sum() * abs(conf - acc)
        ece_per_class.append(ece / len(labels))
    return float(np.mean(ece_per_class)) if ece_per_class else 0.0


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the threshold that maximises macro-F1 across all valid ADR classes.
    Searches the PR curve for each class and averages the per-class optimal thresholds.
    Falls back to label prevalence if sklearn raises an exception.
    """
    from sklearn.metrics import precision_recall_curve
    best_thresholds = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        try:
            prec, rec, thresholds = precision_recall_curve(labels[:, c], scores[:, c])
            if len(thresholds) == 0:
                continue
            f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
            best_thresholds.append(float(thresholds[np.argmax(f1s)]))
        except Exception:
            continue
    if not best_thresholds:
        # ultimate fallback: use label prevalence as threshold
        return float(labels.mean())
    return float(np.mean(best_thresholds))


def compute_f1_mcc(scores: np.ndarray, labels: np.ndarray,
                   threshold: float = None) -> dict:
    """
    Macro-F1, Micro-F1, and MCC using an adaptive threshold.

    threshold : if None (default), the optimal threshold is found by maximising
                macro-F1 on the PR curve of each ADR class and averaging.
                A fixed value (e.g. 0.5) can be forced for comparability.

    The adaptive threshold is essential when label prevalence is low (e.g.
    0.2% density here): a fixed 0.5 threshold will always produce zero
    predictions and zero F1/MCC, masking any learning signal.
    """
    if threshold is None:
        threshold = find_optimal_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return {
        "MacroF1"  : float(f1_score(labels, preds, average="macro",  zero_division=0)),
        "MicroF1"  : float(f1_score(labels, preds, average="micro",  zero_division=0)),
        "MCC"      : float(matthews_corrcoef(labels.ravel(), preds.ravel())),
        "Threshold": float(threshold),   # logged so results are reproducible
    }


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling (post-hoc calibration)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Single-parameter temperature scaling for multi-label classification.
    Divides all logits by a scalar T learned on the validation set.
    T > 1 → model was over-confident (probabilities shrink toward 0.5).
    T < 1 → model was under-confident (probabilities pushed toward 0/1).
    """
    def __init__(self, init_temp: float = 1.5):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * init_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.clamp(min=0.1)


def fit_temperature(model: "PreciseADR", val_graph,
                    device: torch.device,
                    patient_chunk_size: int = 4096) -> TemperatureScaler:
    """
    Fit temperature T on the validation set by minimising binary NLL.
    Model weights are frozen; only T is optimised (LBFGS, 50 iterations).
    Uses chunked inference to avoid OOM on large val sets.
    Returns a CPU TemperatureScaler ready to be passed to evaluate().
    """
    model.eval()
    with torch.no_grad():
        x_dict        = {nt: val_graph[nt].x.to(device)
                         for nt in val_graph.node_types}
        edge_idx_dict = {et: val_graph[et].edge_index.to(device)
                         for et in val_graph.edge_types
                         if hasattr(val_graph[et], "edge_index")}
        mask      = val_graph["patient"].eval_mask.to(device)
        # Chunked GNN forward + predictor
        h_dict    = model.projection(x_dict)
        h_dict    = model.encoder(h_dict, edge_idx_dict)
        h_patient = h_dict["patient"][mask]
        N         = h_patient.shape[0]
        logit_chunks = []
        for start in range(0, N, patient_chunk_size):
            end      = min(start + patient_chunk_size, N)
            h_b      = h_patient[start:end]
            h_aug_b  = model.augment(h_b)
            logit_chunks.append(model.predictor(h_aug_b).cpu())
        logits = torch.cat(logit_chunks, dim=0)

    logits = logits.detach().to(device)
    y      = val_graph["patient"].y[mask.cpu()].float().to(device)

    scaler    = TemperatureScaler(init_temp=1.5).to(device)
    optimizer = torch.optim.LBFGS([scaler.T], lr=0.01, max_iter=50)

    def _closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            scaler(logits), y
        )
        loss.backward()
        return loss

    optimizer.step(_closure)

    T_val = scaler.T.clamp(min=0.1).item()
    direction = "over-confident → scaling down" if T_val > 1.0 else "under-confident → scaling up"
    print(f"  [Temperature Scaling] Fitted T = {T_val:.4f}  ({direction})")
    return scaler.cpu().eval()


def evaluate_all_metrics(scores: np.ndarray, labels: np.ndarray,
                         k_values: list = None) -> dict:
    if k_values is None:
        k_values = EVAL["hit_k"]
    metrics = {
        "AUC"   : compute_auc(scores, labels),
        "AUPRC" : compute_auprc(scores, labels),
        "Brier" : compute_brier_score(scores, labels),
        "ECE"   : compute_ece(scores, labels),
        **compute_f1_mcc(scores, labels),
    }
    for k in k_values:
        metrics[f"Hit@{k}"]    = compute_hit_at_k(scores, labels, k)
        metrics[f"NDCG@{k}"]   = compute_ndcg_at_k(scores, labels, k)
        metrics[f"Recall@{k}"] = compute_recall_at_k(scores, labels, k)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_one_epoch(model: PreciseADR, graph, loss_fn: PreciseADRLoss,
                  optimizer, device: torch.device,
                  is_train: bool = True, scaler=None,
                  accumulation_steps: int = 1,
                  patient_chunk_size: int = 4096,
                  use_two_pass: bool = True) -> dict:
    """
    Training epoch with two modes:

    use_two_pass=True  (memory-safe, slower — for GPUs < 16 GB with FAERS_ALL):
      Pass 1 (torch.no_grad): GNN forward → h_patient. No activations stored.
        Augment/predictor/loss backward runs in chunks accumulating
        dL/d(h_patient) in h_detached.grad.
      Pass 2 (with grad + optional gradient checkpointing): Re-run GNN forward,
        call backward(h_detached.grad) once.
      Peak GPU memory ≈ 1 GB (checkpoint) or 4-6 GB (no checkpoint).
      Cost: 2 GNN forward passes per epoch.

    use_two_pass=False  (single-pass, fastest — for RTX 5090 32 GB):
      Standard single forward → full-batch loss → single backward.
      Peak GPU memory ≈ 4-6 GB (GNN activations for 200k patients).
      Cost: 1 GNN forward pass per epoch (≈ 2× faster than two-pass).
    """
    model.train(is_train)

    x_dict        = {nt: graph[nt].x.to(device) for nt in graph.node_types}
    edge_idx_dict = {et: graph[et].edge_index.to(device)
                     for et in graph.edge_types
                     if hasattr(graph[et], "edge_index")}
    mask = graph["patient"].eval_mask.to(device)
    y    = graph["patient"].y[mask.cpu()].float().to(device)   # (N, n_adrs)
    N    = y.shape[0]

    use_amp = (scaler is not None and is_train)
    amp_ctx = autocast(device_type=device.type, dtype=torch.float16) if use_amp else nullcontext()

    n_chunks    = math.ceil(N / patient_chunk_size)
    total_loss  = 0.0
    total_focal = 0.0
    total_nce   = 0.0

    if is_train:
        optimizer.zero_grad()

        if not use_two_pass:
            # ── Single-pass (RTX 5090 / ≥16 GB): standard forward + backward ──
            # Full batch through GNN → augment → predictor → loss → backward.
            # Peak memory: GNN layer activations (~4-6 GB) + logits (~350 MB fp16).
            # ~2× faster than two-pass because there is only one GNN forward.
            with amp_ctx:
                h_patient = model.embed_patients(x_dict, edge_idx_dict, mask)
                h_aug     = model.augment(h_patient)
                logits    = model.predictor(h_aug)
                loss, focal, nce = loss_fn(logits, y, h_patient, h_aug)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss  = loss.item()
            total_focal = focal.item()
            total_nce   = nce.item()

        else:
            # ── Two-pass (small GPU / OOM-safe) ──────────────────────────────

            # Pass 1: GNN forward with no_grad – no activation tensors stored.
            with torch.no_grad(), amp_ctx:
                h_patient = model.embed_patients(x_dict, edge_idx_dict, mask)

            # Detach so augment/predictor/loss graph is independent of GNN.
            h_detached = h_patient.detach().requires_grad_(True)
            del h_patient   # free; pass 2 will recompute it

            # Chunked loss backward (augment + predictor only).
            # No retain_graph: GNN is not in the computation graph here.
            for start in range(0, N, patient_chunk_size):
                end = min(start + patient_chunk_size, N)
                with amp_ctx:
                    h_b      = h_detached[start:end]
                    y_b      = y[start:end]
                    h_aug_b  = model.augment(h_b)
                    logits_b = model.predictor(h_aug_b)
                    loss_b, focal_b, nce_b = loss_fn(logits_b, y_b, h_b, h_aug_b)
                    loss_b   = loss_b / n_chunks

                if scaler is not None:
                    scaler.scale(loss_b).backward()
                else:
                    loss_b.backward()

                total_loss  += loss_b.item() * n_chunks
                total_focal += focal_b.item()
                total_nce   += nce_b.item()

            # Pass 2: GNN backward via one checkpointed re-forward.
            # h_detached.grad holds dL/d(h_patient) accumulated from all chunks.
            if h_detached.grad is not None:
                with amp_ctx:
                    h_patient2 = model.embed_patients(x_dict, edge_idx_dict, mask)
                h_patient2.backward(h_detached.grad)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    else:
        # Eval: loss metrics only (no backward), chunked to avoid OOM.
        with torch.no_grad():
            h_patient = model.embed_patients(x_dict, edge_idx_dict, mask)
            for start in range(0, N, patient_chunk_size):
                end      = min(start + patient_chunk_size, N)
                h_b      = h_patient[start:end]
                y_b      = y[start:end]
                h_aug_b  = model.augment(h_b)
                logits_b = model.predictor(h_aug_b)
                loss_b, focal_b, nce_b = loss_fn(logits_b, y_b, h_b, h_aug_b)
                total_loss  += loss_b.item()
                total_focal += focal_b.item()
                total_nce   += nce_b.item()

    # Normalise: two-pass and eval accumulate per-chunk means; single-pass is already global mean.
    divisor = 1 if (is_train and not use_two_pass) else n_chunks
    return {
        "loss"   : total_loss  / divisor,
        "focal"  : total_focal / divisor,
        "infonce": total_nce   / divisor,
    }


def evaluate(model: PreciseADR, graph, device: torch.device,
             temp_scaler: TemperatureScaler = None,
             patient_chunk_size: int = 4096) -> dict:
    """
    Compute all metrics on a graph split.

    GNN forward runs on the full graph; prediction is chunked to avoid
    OOM on large datasets (e.g. FAERS_ALL).
    If temp_scaler is provided, logits are temperature-scaled before sigmoid.
    """
    model.eval()
    with torch.no_grad():
        x_dict        = {nt: graph[nt].x.to(device) for nt in graph.node_types}
        edge_idx_dict = {et: graph[et].edge_index.to(device)
                         for et in graph.edge_types
                         if hasattr(graph[et], "edge_index")}
        mask = graph["patient"].eval_mask.to(device)
        y    = graph["patient"].y[mask.cpu()].numpy()

        # Full GNN forward
        h_dict    = model.projection(x_dict)
        h_dict    = model.encoder(h_dict, edge_idx_dict)
        h_patient = h_dict["patient"][mask]    # (N, D)
        N         = h_patient.shape[0]

        # Chunked predictor → sigmoid (avoids [N, n_adrs] OOM)
        scores_chunks = []
        for start in range(0, N, patient_chunk_size):
            end      = min(start + patient_chunk_size, N)
            h_b      = h_patient[start:end]
            h_aug_b  = model.augment(h_b)       # dropout off in eval mode
            logits_b = model.predictor(h_aug_b)
            if temp_scaler is not None:
                logits_b = temp_scaler.to(device)(logits_b)
            scores_chunks.append(torch.sigmoid(logits_b).cpu())

        scores = torch.cat(scores_chunks, dim=0).numpy()

    return evaluate_all_metrics(scores, y)


def compute_pos_weight(train_graph,
                       cap: float = 50.0) -> torch.Tensor:
    """
    Compute per-ADR positive class weights from training labels.

    Formula: w_c = (N_neg_c / N_pos_c), clipped to [1, cap].
    ADRs with more negatives than positives get upweighted; the cap prevents
    extreme values for very rare ADRs (e.g., 2 positives in 10k patients → w=5k).

    Returns a (n_adrs,) CPU float tensor ready to be passed to FocalLoss.
    """
    mask   = train_graph["patient"].eval_mask          # training patient mask
    y      = train_graph["patient"].y[mask].float()    # (N_train, N_adrs)
    pos    = y.sum(dim=0).clamp(min=1)
    neg    = (y.shape[0] - y.sum(dim=0)).clamp(min=1)
    weight = (neg / pos).clamp(min=1.0, max=cap)
    print(f"  [pos_weight] min={weight.min():.1f}  "
          f"mean={weight.mean():.1f}  "
          f"max={weight.max():.1f}  "
          f"(cap={cap}, n_adrs={weight.shape[0]})")
    return weight.cpu()


def train_model(train_graph, val_graph, model: PreciseADR,
                alpha: float, tau: float, gamma: float,
                lr: float, max_epochs: int, patience: int,
                device: torch.device, use_amp: bool = True,
                accumulation_steps: int = 1,
                pos_weight: torch.Tensor = None,
                weight_decay: float = 0.0,
                patient_chunk_size: int = 4096,
                use_two_pass: bool = True,
                use_lr_scheduler: bool = False) -> tuple[PreciseADR, dict, dict]:
    """
    Train PreciseADR and return (best_model, best_val_metrics, history).

    Args:
        use_amp             : Use automatic mixed precision (FP16) training.
        accumulation_steps  : Number of gradient accumulation steps.
        pos_weight          : (n_adrs,) per-class positive weights for FocalLoss.
        patient_chunk_size  : Patients processed per chunk in eval/two-pass loss.
        use_two_pass        : Two-pass training (OOM-safe); False = single-pass
                              (requires ~4-6 GB GPU, ~2× faster).
        use_lr_scheduler    : ReduceLROnPlateau on val_AUC (factor=0.5, patience=20).
    """
    # Reset seed for reproducibility if seed was set
    if _GLOBAL_SEED is not None:
        set_seed(_GLOBAL_SEED, verbose=False)

    model   = model.to(device)
    loss_fn = PreciseADRLoss(alpha=alpha, tau=tau, gamma=gamma,
                              pos_weight=pos_weight).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize gradient scaler for mixed precision training
    amp_scaler = GradScaler() if (use_amp and device.type == 'cuda') else None

    # LR scheduler: halve LR after 20 epochs with no val_AUC improvement.
    # min_lr=1e-5 prevents LR from collapsing to zero.
    scheduler = None
    if use_lr_scheduler:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                      patience=20, min_lr=1e-5, verbose=False)

    best_auc      = -1.0
    best_state    = None
    patience_left = patience

    history = {"train_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, max_epochs + 1):
        train_info = run_one_epoch(model, train_graph, loss_fn, optimizer,
                                   device, is_train=True, scaler=amp_scaler,
                                   accumulation_steps=accumulation_steps,
                                   patient_chunk_size=patient_chunk_size,
                                   use_two_pass=use_two_pass)
        val_metrics = evaluate(model, val_graph, device,
                               patient_chunk_size=patient_chunk_size)
        val_auc     = val_metrics["AUC"]
        cur_lr      = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_info["loss"])
        history["val_auc"].append(val_auc)
        history["lr"].append(cur_lr)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1

        if scheduler is not None:
            scheduler.step(val_auc)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < cur_lr:
                print(f"  [LR] reduced {cur_lr:.2e} → {new_lr:.2e} at epoch {epoch}")

        if epoch % 10 == 0 or patience_left == 0:
            print(f"  Epoch {epoch:3d} | loss={train_info['loss']:.4f} "
                  f"focal={train_info['focal']:.4f} nce={train_info['infonce']:.4f} "
                  f"val_AUC={val_auc:.4f} | best={best_auc:.4f}  lr={cur_lr:.2e}")

        if patience_left == 0:
            print(f"  Early stopping at epoch {epoch}")
            break

        # Clear cache after each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    model.load_state_dict(best_state)
    best_val_metrics = evaluate(model, val_graph, device,
                                patient_chunk_size=patient_chunk_size)
    return model, best_val_metrics, history


# ─────────────────────────────────────────────────────────────────────────────
# Grid search over α × τ
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(train_graph, val_graph, base_model_cfg: dict,
                alpha_range: list, tau_range: list,
                device: torch.device, out_dir: str,
                pos_weight: torch.Tensor = None,
                weight_decay: float = 0.0) -> tuple[float, float]:
    """
    Perform α × τ grid search on validation AUC.
    Returns (best_alpha, best_tau).
    Saves a 2-D results table as grid_search_results.json.
    pos_weight is forwarded to every train_model call so the grid search
    is evaluated under the same loss as the final training run.
    """
    print(f"\n  Grid search: {len(alpha_range)} α × {len(tau_range)} τ "
          f"= {len(alpha_range)*len(tau_range)} runs"
          f"  [pos_weight={'on' if pos_weight is not None else 'off'}]")
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
            pos_weight=pos_weight,
            weight_decay=weight_decay,
            patient_chunk_size=base_model_cfg.get("patient_chunk_size", 4096),
            use_two_pass=base_model_cfg.get("use_two_pass", True),
            use_lr_scheduler=False,   # no scheduler during grid search (short runs)
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
                  do_grid_search: bool, device: torch.device,
                  out_suffix: str = ""):
    """
    out_suffix : appended to the per-variant result sub-directory and checkpoint
                 filename so different runs (e.g. gamma1 vs gamma2) don't overwrite
                 each other.  Example: out_suffix="_gamma2" → xxx_gamma2/.
    """
    suffix_label = f" [{out_suffix.lstrip('_')}]" if out_suffix else ""
    print(f"\n{'='*70}")
    print(f"  TRAINING: {variant.upper()}{suffix_label}  "
          f"γ={cfg['focal_gamma']}  α={cfg['alpha']}  τ={cfg['tau']}")
    print(f"{'='*70}")

    var_graph_dir = os.path.join(graph_path, variant)
    out_dir       = os.path.join(results_path, f"{variant}{out_suffix}")
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

    # Per-ADR positive weighting (Option C / REC-12)
    pos_weight = None
    if cfg.get("use_pos_weight", False):
        pos_weight = compute_pos_weight(
            train_graph, cap=cfg.get("pos_weight_cap", 50.0)
        )

    # Grid search
    if do_grid_search:
        best_alpha, best_tau = grid_search(
            train_graph, val_graph,
            base_model_cfg=cfg,
            alpha_range=cfg["alpha_range"],
            tau_range=cfg["tau_range"],
            device=device,
            out_dir=out_dir,
            pos_weight=pos_weight,
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    else:
        best_alpha = cfg["alpha"]
        best_tau   = cfg["tau"]

    # Full training with best hyperparameters
    print(f"\n  Full training: α={best_alpha}  τ={best_tau}"
          f"  pos_weight={'on' if pos_weight is not None else 'off'}")

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
        pos_weight=pos_weight,
        weight_decay=cfg.get("weight_decay", 0.0),
        patient_chunk_size=cfg.get("patient_chunk_size", 4096),
        use_two_pass=cfg.get("use_two_pass", True),
        use_lr_scheduler=cfg.get("use_lr_scheduler", False),
    )

    # ── Temperature scaling (post-hoc calibration on val set) ────────────────
    temp_scaler = fit_temperature(model, val_graph, device,
                                  patient_chunk_size=cfg.get("patient_chunk_size", 4096))

    # ── Test evaluation (calibrated) ─────────────────────────────────────────
    chunk = cfg.get("patient_chunk_size", 4096)
    test_metrics_uncal = evaluate(model, test_graph, device,
                                  patient_chunk_size=chunk)
    test_metrics       = evaluate(model, test_graph, device,
                                  temp_scaler=temp_scaler,
                                  patient_chunk_size=chunk)

    # ── Print results ─────────────────────────────────────────────────────────
    _RANK_METRICS = {"AUC", "AUPRC", "Hit@1", "Hit@2", "Hit@5", "Hit@10", "Hit@20",
                     "NDCG@1", "NDCG@2", "NDCG@5", "NDCG@10", "NDCG@20",
                     "Recall@1", "Recall@2", "Recall@5", "Recall@10", "Recall@20"}
    _CAL_METRICS  = {"Brier", "ECE", "MacroF1", "MicroF1", "MCC"}

    print(f"\n  Validation metrics (uncalibrated):")
    for k, v in val_metrics.items():
        print(f"    {k}: {v:.4f}")

    print(f"\n  Test metrics — ranking (calibrated with T):")
    for k, v in test_metrics.items():
        if k in _RANK_METRICS:
            uncal = test_metrics_uncal.get(k, float("nan"))
            print(f"    {k}: {v:.4f}  (uncal: {uncal:.4f})")

    print(f"\n  Test metrics — calibration / classification (T-scaled):")
    for k in ("Brier", "ECE", "MacroF1", "MicroF1", "MCC"):
        v     = test_metrics.get(k, float("nan"))
        uncal = test_metrics_uncal.get(k, float("nan"))
        print(f"    {k}: {v:.4f}  (uncal: {uncal:.4f})")

    # Save model checkpoint (includes temperature scaler state)
    ckpt_name = f"{variant}{out_suffix}_best_infonce.pt"
    ckpt_path = os.path.join(model_path, ckpt_name)
    torch.save({
        "model_state_dict"  : model.state_dict(),
        "temperature_scaler": temp_scaler.state_dict(),
        "temperature_T"     : temp_scaler.T.item(),
        "focal_gamma"       : cfg["focal_gamma"],
        "alpha"             : best_alpha,
        "tau"               : best_tau,
        "val_metrics"       : val_metrics,
        "test_metrics_cal"  : test_metrics,
        "test_metrics_uncal": test_metrics_uncal,
        "history"           : history,
    }, ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}")

    # Save metrics JSON
    metrics_path = os.path.join(out_dir, "final_metrics_infonce.json")
    pw_stats = ({"min": float(pos_weight.min()), "mean": float(pos_weight.mean()),
                 "max": float(pos_weight.max())}
                if pos_weight is not None else None)
    metrics_data = {
        "variant"           : variant,
        "focal_gamma"       : float(cfg["focal_gamma"]),
        "alpha"             : float(best_alpha),
        "tau"               : float(best_tau),
        "use_pos_weight"    : pos_weight is not None,
        "pos_weight_stats"  : pw_stats,
        "temperature_T"     : float(temp_scaler.T.item()),
        "val_metrics"       : convert_to_serializable(val_metrics),
        "test_metrics_cal"  : convert_to_serializable(test_metrics),
        "test_metrics_uncal": convert_to_serializable(test_metrics_uncal),
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
    p.add_argument("--focal-gamma",    type=float, default=None,
                   help="Override config focal_gamma (e.g. --focal-gamma 2.0). "
                        "Useful for Option B comparison without editing config.py.")
    p.add_argument("--out-suffix",     type=str,   default="",
                   help="Suffix appended to results/model sub-directories "
                        "(e.g. --out-suffix _gamma2). Prevents overwriting previous runs.")
    p.add_argument("--no-grid-search", action="store_true")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int, default=123,
                   help="Random seed for reproducibility (default: None, no seed set)")
    # ── KG ablation overrides (override config.py flags at runtime) ──────────
    p.add_argument("--use-onsides-adr", action="store_true", default=None,
                   help="Enable OnSIDES drug-label ADE edges (onsides_has_adr). "
                        "Overrides config.py KG_EDGES['use_onsides_adr']. "
                        "Requires graphs already built by step2 with use_onsides_adr=True.")
    p.add_argument("--no-onsides-adr",  dest="use_onsides_adr", action="store_false",
                   help="Disable OnSIDES ADE edges (override config.py).")
    p.set_defaults(use_onsides_adr=None)  # None = respect config.py
    # ── Per-ADR positive weight (Option C / REC-12) ───────────────────────────
    p.add_argument("--pos-weight",    dest="use_pos_weight", action="store_true",
                   default=None,
                   help="Enable per-ADR positive weighting in FocalLoss "
                        "(neg_count/pos_count, capped at pos_weight_cap). "
                        "Overrides config.py MODEL['use_pos_weight'].")
    p.add_argument("--no-pos-weight", dest="use_pos_weight", action="store_false",
                   help="Disable per-ADR positive weighting (override config.py).")
    p.set_defaults(use_pos_weight=None)  # None = respect config.py
    # ── Training mode ──────────────────────────────────────────────────────────
    p.add_argument("--two-pass",    dest="use_two_pass", action="store_true",
                   default=None,
                   help="Force two-pass training (OOM-safe for small GPUs). "
                        "Overrides config.py MODEL['use_two_pass'].")
    p.add_argument("--no-two-pass", dest="use_two_pass", action="store_false",
                   help="Force single-pass training (faster, needs ≥16 GB GPU). "
                        "Overrides config.py MODEL['use_two_pass'].")
    p.set_defaults(use_two_pass=None)  # None = respect config.py
    p.add_argument("--deterministic", dest="deterministic", action="store_true",
                   default=False,
                   help="Enable cuDNN deterministic mode (bit-exact reproducibility "
                        "but 10-30%% slower). Default: off (seeded but non-deterministic).")
    return p.parse_args()


def main():
    args   = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        set_seed(args.seed, deterministic=args.deterministic)

    # ── Apply KG runtime overrides before graph loading ───────────────────────
    if args.use_onsides_adr is not None:
        _cfg_module.KG_EDGES["use_onsides_adr"] = args.use_onsides_adr
        print(f"  [CLI override] KG_EDGES['use_onsides_adr'] = {args.use_onsides_adr}")

    device = torch.device(args.device)
    cfg    = dict(MODEL)   # copy from config

    if args.alpha       is not None: cfg["alpha"]       = args.alpha
    if args.tau         is not None: cfg["tau"]         = args.tau
    if args.focal_gamma is not None:
        cfg["focal_gamma"] = args.focal_gamma
        print(f"  [CLI override] focal_gamma = {args.focal_gamma}")
    if args.use_pos_weight is not None:
        cfg["use_pos_weight"] = args.use_pos_weight
        print(f"  [CLI override] use_pos_weight = {args.use_pos_weight}")
    if args.use_two_pass is not None:
        cfg["use_two_pass"] = args.use_two_pass
        print(f"  [CLI override] use_two_pass = {args.use_two_pass}")

    do_grid    = not args.no_grid_search
    out_suffix = args.out_suffix or ""

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
            out_suffix=out_suffix,
        )
        all_results[variant] = metrics

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for v, m in all_results.items():
        print(f"  {v:15s}  AUC={m['AUC']:.4f}  AUPRC={m.get('AUPRC', 0):.4f}"
              f"  Hit@10={m.get('Hit@10', 0):.4f}"
              f"  Brier={m.get('Brier', 0):.4f}  ECE={m.get('ECE', 0):.4f}"
              f"  F1={m.get('MacroF1', 0):.4f}  MCC={m.get('MCC', 0):.4f}")

    os.makedirs(args.results_path, exist_ok=True)
    with open(os.path.join(args.results_path, "training_summary_infonce.json"), "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    print("\n✓ Step 4 (Training) complete!")


if __name__ == "__main__":
    main()
