"""
step6c_ablation_demographic.py
================================
Ablation Category 3 (Section 6.3): Demographic Feature Sensitivity

Research question: How important are gender and age features for
predicting gender-related and age-related ADRs?

Protocol:
  - Train the full PreciseADR model once (no perturbation).
  - At inference time, apply perturbations to test-patient node features:
      Full          : no perturbation
      Disturb Gender: replace gender with random M/F for each test patient
      Disturb Age   : replace age category with random Youth/Adult/Elderly
      Disturb Both  : replace both gender and age with random values
  - Evaluate on:
      (a) 20 gender-related ADRs  (replicating Figures 3a, 4a/c)
      (b) 20 age-related ADRs     (replicating Figures 3b, 4b/d)
  - Report per-ADR AUC heatmaps and aggregate metrics.
  - Calculate % AUC reduction vs. full model.

Output:
  results/{variant}/ablation_demographic.json
  results/{variant}/ablation_demographic_auc_heatmap.csv

Usage
-----
    python step6c_ablation_demographic.py [--variant xxx]
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import GRAPH_PATH, MODEL_PATH, RESULTS_PATH, PREPROCESSED_PATH, MODEL, ABLATION
from config import DATASET_NAME, OUTPUT_PATH
from step3_model_supcon import build_model_from_graph, PreciseADRLoss
from step4_training_supcon import evaluate_all_metrics, train_model, convert_to_serializable

METHOD = "supcon"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_supcon")

# ─────────────────────────────────────────────────────────────────────────────
# Feature perturbation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _perturb_patient_features(x: torch.Tensor, mask: torch.Tensor,
                               perturb_gender: bool = False,
                               perturb_age: bool    = False,
                               rng: np.random.Generator = None) -> torch.Tensor:
    """
    Patient feature layout (7 dims):
      [0,1,2]  : age one-hot (Youth, Adult, Elderly)
      [3,4]    : gender one-hot (Male, Female)
      [5]      : num_drugs_normed
      [6]      : num_diseases_normed

    mask : boolean mask selecting test patients in the full patient tensor.
    """
    if rng is None:
        rng = np.random.default_rng(args.seed)

    x_perturbed = x.clone()
    eval_idx    = torch.where(mask)[0]
    n_eval      = len(eval_idx)

    if perturb_age:
        # Random age bin (uniform over Youth/Adult/Elderly)
        rand_age_idx = rng.integers(0, 3, size=n_eval)
        new_age = np.eye(3)[rand_age_idx]          # (n_eval, 3)
        x_perturbed[eval_idx, 0:3] = torch.tensor(new_age, dtype=torch.float)

    if perturb_gender:
        rand_gender_idx = rng.integers(0, 2, size=n_eval)
        new_gender = np.eye(2)[rand_gender_idx]    # (n_eval, 2)
        x_perturbed[eval_idx, 3:5] = torch.tensor(new_gender, dtype=torch.float)

    return x_perturbed


# ─────────────────────────────────────────────────────────────────────────────
# Per-ADR AUC helper
# ─────────────────────────────────────────────────────────────────────────────

def auc_per_adr(scores: np.ndarray, labels: np.ndarray,
                adr_indices: list) -> dict:
    """
    Compute AUC for each selected ADR index.
    Returns {adr_idx: auc_value}.
    """
    result = {}
    for idx in adr_indices:
        y_col = labels[:, idx]
        s_col = scores[:, idx]
        if y_col.sum() == 0 or (1 - y_col).sum() == 0:
            result[idx] = np.nan
        else:
            try:
                result[idx] = float(roc_auc_score(y_col, s_col))
            except Exception:
                result[idx] = np.nan
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Load related ADR indices
# ─────────────────────────────────────────────────────────────────────────────

def _build_adr_name_to_code_mapping():
    """
    Build a mapping from ADR names to MedDRA codes using the filtered ADR table.
    Returns: dict mapping ADR name (str) → MedDRA code (str)
    """
    ar_path = os.path.join(PREPROCESSED_PATH, "ADVERSE_REACTIONS_FILTERED.csv")
    if not os.path.exists(ar_path):
        print(f"  Warning: {ar_path} not found, cannot build name→code mapping")
        return {}
    
    ar_df = pd.read_csv(ar_path, sep=',')
    # Use PT_CODE (preferred term code) as the MedDRA code
    name_to_code = dict(zip(ar_df['ADVERSE_EVENT'], ar_df['PT_CODE'].astype(str)))
    return name_to_code


def _load_related_adr_indices(adr_vocab: dict, related_adrs: list,
                               n_top: int = 20) -> list:
    """
    Map a list of ADR name/code strings to column indices in the label matrix.
    
    Args:
        adr_vocab: dict mapping MedDRA codes (str) → column index (int)
        related_adrs: list of ADR names (from gender/age-related ADRs JSON)
        n_top: maximum number of indices to return
    
    Returns:
        List of column indices for related ADRs found in the vocabulary
    """
    # Build name→code mapping
    name_to_code = _build_adr_name_to_code_mapping()
    
    indices = []
    for adr_name in related_adrs:
        if len(indices) >= n_top:
            break
        
        # Try direct lookup first (in case it's already a code)
        if adr_name in adr_vocab:
            indices.append(adr_vocab[adr_name])
            continue
        
        # Convert name to code and lookup
        adr_code = name_to_code.get(adr_name)
        if adr_code and adr_code in adr_vocab:
            indices.append(adr_vocab[adr_code])
    
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Main ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_demographic_ablation(variant: str, graph_path: str,
                              results_path: str, model_path: str,
                              device: torch.device, seed: int=123):
    print(f"\n{'='*70}")
    print(f"  ABLATION 6.3 – DEMOGRAPHIC SENSITIVITY: {variant.upper()}")
    print(f"{'='*70}")

    graph_dir = os.path.join(graph_path, variant)
    out_dir   = os.path.join(results_path, variant)
    os.makedirs(out_dir, exist_ok=True)

    train_graph = torch.load(os.path.join(graph_dir, "train_graph.pt"), weights_only=False)
    val_graph   = torch.load(os.path.join(graph_dir, "val_graph.pt"), weights_only=False)
    test_graph  = torch.load(os.path.join(graph_dir, "test_graph.pt"), weights_only=False)

    # Load or train the full model
    ckpt_path = os.path.join(model_path, f"{variant}_best_supcon.pt")
    cfg = dict(MODEL)
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_graph(train_graph, cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        cfg["alpha"] = ckpt.get("alpha", cfg["alpha"])
        cfg["tau"]   = ckpt.get("tau",   cfg["tau"])
        print("  Loaded pre-trained model.")
    else:
        print("  No checkpoint found – training from scratch …")
        model, _, _ = train_model(
            train_graph, val_graph, build_model_from_graph(train_graph, cfg),
            alpha=cfg["alpha"], tau=cfg["tau"], gamma=cfg["focal_gamma"],
            lr=cfg["lr"], max_epochs=cfg["max_epochs"], patience=cfg["patience"],
            device=device)

    model = model.to(device)
    model.eval()

    # Load related ADRs
    adr_vocab = train_graph.adr_vocab
    # Use PREPROCESSED_PATH from config for correct location
    gender_path = os.path.join(PREPROCESSED_PATH, "gender_related_adrs.json")
    age_path    = os.path.join(PREPROCESSED_PATH, "age_related_adrs.json")

    # Load gender-related ADRs
    if os.path.exists(gender_path):
        with open(gender_path) as f:
            gender_adrs_raw = json.load(f)
    else:
        print(f"  Warning: {gender_path} not found")
        gender_adrs_raw = []

    # Load age-related ADRs
    if os.path.exists(age_path):
        with open(age_path) as f:
            age_adrs_raw = json.load(f)
    else:
        print(f"  Warning: {age_path} not found")
        age_adrs_raw = []



    n_sel  = ABLATION.get("n_gender_adrs", 20)
    gender_idx = _load_related_adr_indices(adr_vocab, gender_adrs_raw, n_sel)
    age_idx    = _load_related_adr_indices(adr_vocab, age_adrs_raw,    n_sel)
    print(f"  Gender-related ADRs for evaluation: {len(gender_idx)}")
    print(f"  Age-related ADRs for evaluation:    {len(age_idx)}")

    # ── Perturbation loop ─────────────────────────────────────────────────
    perturbations = {
        "Full"           : (False, False),
        "Disturb_Gender" : (True,  False),
        "Disturb_Age"    : (False, True),
        "Disturb_Both"   : (True,  True),
    }

    all_results   = {}
    heatmap_rows  = []
    rng           = np.random.default_rng(seed)

    for name, (pg, pa) in perturbations.items():
        print(f"\n  Perturbation: {name}")

        # Modify patient node features
        x_orig  = test_graph["patient"].x.clone()
        x_mod   = _perturb_patient_features(
            x_orig, test_graph["patient"].eval_mask,
            perturb_gender=pg, perturb_age=pa, rng=rng)

        # Forward pass with modified features
        with torch.no_grad():
            x_dict  = {nt: test_graph[nt].x.to(device) for nt in test_graph.node_types}
            x_dict["patient"] = x_mod.to(device)
            ei_dict = {et: test_graph[et].edge_index.to(device)
                       for et in test_graph.edge_types
                       if hasattr(test_graph[et], "edge_index")}
            tmask   = test_graph["patient"].eval_mask.to(device)
            logits, _, _, _ = model(x_dict, ei_dict, patient_mask=tmask)
            scores  = torch.sigmoid(logits).cpu().numpy()

        y_true = test_graph["patient"].y[test_graph["patient"].eval_mask].numpy()

        # Per-ADR AUC
        g_auc = auc_per_adr(scores, y_true, gender_idx)
        a_auc = auc_per_adr(scores, y_true, age_idx)

        mean_g = float(np.nanmean(list(g_auc.values()))) if g_auc else 0.0
        mean_a = float(np.nanmean(list(a_auc.values()))) if a_auc else 0.0

        print(f"    Mean AUC (gender-ADRs): {mean_g:.4f}")
        print(f"    Mean AUC (age-ADRs):    {mean_a:.4f}")

        all_results[name] = {
            "gender_adr_auc_per_idx" : {str(k): v for k, v in g_auc.items()},
            "age_adr_auc_per_idx"    : {str(k): v for k, v in a_auc.items()},
            "mean_gender_adr_auc"    : mean_g,
            "mean_age_adr_auc"       : mean_a,
        }

        heatmap_rows.append({
            "perturbation"      : name,
            "mean_gender_auc"   : mean_g,
            "mean_age_auc"      : mean_a,
        })

    # Compute % AUC reduction vs. Full model
    full_g = all_results["Full"]["mean_gender_adr_auc"]
    full_a = all_results["Full"]["mean_age_adr_auc"]
    for name, res in all_results.items():
        if name == "Full":
            res["pct_reduction_gender"] = 0.0
            res["pct_reduction_age"]    = 0.0
        else:
            res["pct_reduction_gender"] = (
                (full_g - res["mean_gender_adr_auc"]) / full_g * 100
                if full_g > 0 else 0.0)
            res["pct_reduction_age"] = (
                (full_a - res["mean_age_adr_auc"]) / full_a * 100
                if full_a > 0 else 0.0)

    out_json = os.path.join(out_dir, "ablation_demographic_supcon.json")
    with open(out_json, "w") as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    pd.DataFrame(heatmap_rows).to_csv(
        os.path.join(out_dir, "ablation_demographic_auc_heatmap_supcon.csv"), index=False)

    print(f"\n  Saved → {out_json}")
    print(f"  % AUC reduction (Full→Disturb_Both): "
          f"gender={all_results['Disturb_Both']['pct_reduction_gender']:.1f}%  "
          f"age={all_results['Disturb_Both']['pct_reduction_age']:.1f}%")
    return all_results


def parse_args():
    p = argparse.ArgumentParser(description="Ablation 6.3: Demographic Sensitivity")
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
    for v in variants:
        if os.path.exists(os.path.join(args.graph_path, v)):
            run_demographic_ablation(v, args.graph_path, args.results_path,
                                     args.model_path, device, args.seed)
    print("\n✓ Ablation 6.3 (Demographic Sensitivity) complete!")


if __name__ == "__main__":
    main()
