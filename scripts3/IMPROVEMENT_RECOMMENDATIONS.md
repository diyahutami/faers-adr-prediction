# Recommendations for Improving ADR Prediction in scripts3
## Based on Literature Review: Explainability, Calibration, and Performance

**Project:** TB ADR Prediction using HGNN on FAERS (scripts3 pipeline)  
**Literature basis:** `Literature_Review_ADR_Explainability.md` [R1–R6] and `Literature_Review_AI_Implementation_ADR.md` [I1–I10]  
**Date:** 2026-04-13

---

## Overview of Current Code State

| Aspect | Current State | Gap |
|---|---|---|
| Primary metric | AUC-ROC (macro) | No AUPRC, no F1, no MCC |
| Calibration | None | Brier/ECE missing; mandatory per [R4, R6] |
| Patient features | 7-dim (age, gender, n_drugs, n_diseases) | Missing country, reporter type, outcome severity |
| Drug features | One-hot identity matrix (`n_drugs × n_drugs`) | No molecular/structural features; drugs in same class share nothing |
| ADR features | One-hot identity matrix (`n_adrs × n_adrs`) | No semantic similarity; "hepatotoxicity" ≡ "optic neuritis" to model |
| Disease features | One-hot identity matrix (`n_dis × n_dis`) | No ontology embeddings |
| Focal loss gamma | 1.0 | Original paper = 2.0; needs re-testing on new dataset |
| InfoNCE alpha | 0.0 (default config) | Contrastive learning disabled by default |
| Train-test split | Random (75/12.5/12.5) | PERIOD column available but unused for temporal split |
| Explainability | None | No SHAP, no GNNExplainer, no counterfactuals |
| Fairness audit | None | No subgroup evaluation by age, gender, country |

---

## Tier 1 — Quick Wins (30 minutes to 4 hours each)

---

### REC-1 | Add AUPRC to Every Evaluation Call

**Files to change:** `scripts3/step4_training_infonce.py`  
**Effort:** 30 minutes  
**Evidence:** [R4], [I5] — AUROC is misleading with rare events. AUPRC directly measures precision-recall trade-off, which is more clinically relevant when false negatives are costly.

**Current state:** `evaluate_all_metrics()` computes only AUC-ROC, Hit@K, NDCG@K, Recall@K. No AUPRC.

**Change:**
```python
from sklearn.metrics import average_precision_score

# In evaluate_all_metrics():
valid = labels.sum(axis=0) > 0
metrics["AUPRC"] = average_precision_score(
    labels[:, valid], scores[:, valid], average="macro"
)
```

**Expected outcome:** No performance gain — this is a measurement gap. AUPRC provides a more honest picture for imbalanced ADR labels and is required for most clinical AI publications.

---

### REC-2 | Add Brier Score and Expected Calibration Error (ECE)

**Files to change:** `scripts3/step4_training_infonce.py`  
**Effort:** 2 hours  
**Evidence:** [R4], [R6] identify calibration as mandatory for clinical deployment. NFR2 specifies ECE < 0.05 as a deployment threshold. Zero papers in the 10-paper implementation review report any calibration metric — an immediate gap to fill.

**Current state:** No calibration metrics anywhere in scripts3.

**Change:**
```python
from sklearn.metrics import brier_score_loss

def _compute_ece(scores, labels, n_bins=10):
    """Expected Calibration Error across all ADR classes."""
    ece_per_class = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(scores[:, c], bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            conf = scores[:, c][mask].mean()
            acc  = labels[:, c][mask].mean()
            ece += mask.sum() * abs(conf - acc)
        ece_per_class.append(ece / len(labels))
    return float(np.mean(ece_per_class)) if ece_per_class else 0.0

# In evaluate_all_metrics():
brier_scores = [
    brier_score_loss(labels[:, c], scores[:, c])
    for c in range(labels.shape[1]) if labels[:, c].sum() > 0
]
metrics["Brier"] = float(np.mean(brier_scores))
metrics["ECE"]   = _compute_ece(scores, labels, n_bins=10)
```

**Expected outcome:** Exposes whether the model is systematically over- or under-confident. ECE target < 0.05 for clinical deployment. Without this, probabilities cannot be interpreted as actual risk percentages.

---

### REC-3 | Add Temperature Scaling (Post-hoc Calibration)

**Files to change:** `scripts3/step4_training_infonce.py`  
**Effort:** 3 hours  
**Evidence:** [R4] — GNNs produce poorly calibrated logits by default. Temperature scaling is the simplest and most effective post-hoc fix (one scalar learned on the validation set after training).

**Current state:** Logits go directly to sigmoid with no calibration step.

**Change:**
```python
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.T.clamp(min=0.1)

def fit_temperature(model, val_graph, device):
    """Learn temperature T by minimising NLL on validation set."""
    scaler = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([scaler.T], lr=0.01, max_iter=50)
    model.eval()
    with torch.no_grad():
        x_dict    = {nt: val_graph[nt].x.to(device) for nt in val_graph.node_types}
        edge_dict = {et: val_graph[et].edge_index.to(device)
                     for et in val_graph.edge_types if hasattr(val_graph[et], "edge_index")}
        mask   = val_graph["patient"].eval_mask.to(device)
        logits, _, _ = model(x_dict, edge_dict, patient_mask=mask)
        y      = val_graph["patient"].y[mask.cpu()].float().to(device)

    def eval_nll():
        optimizer.zero_grad()
        scaled = scaler(logits)
        loss = nn.functional.binary_cross_entropy_with_logits(scaled, y)
        loss.backward()
        return loss
    optimizer.step(eval_nll)
    return scaler

# After train_model(): fit T on val, apply to test logits
```

**Expected outcome:** Typically reduces ECE by 30–60% with no accuracy loss. Essential for credible risk probability outputs.

---

### REC-4 | Add MCC and Macro-F1 to Evaluation

**Files to change:** `scripts3/step4_training_infonce.py`  
**Effort:** 1 hour  
**Evidence:** [I5] uses MCC as primary metric for imbalanced data. [R4] requires sensitivity/specificity at chosen threshold. F1 is more interpretable by clinical collaborators than AUC.

**Current state:** No F1, no MCC, no threshold-dependent metrics.

**Change:**
```python
from sklearn.metrics import f1_score, matthews_corrcoef

# In evaluate_all_metrics() after AUC and AUPRC:
preds = (scores > 0.5).astype(int)
metrics["MacroF1"] = f1_score(labels, preds, average="macro", zero_division=0)
metrics["MicroF1"] = f1_score(labels, preds, average="micro", zero_division=0)
metrics["MCC"]     = matthews_corrcoef(labels.ravel(), preds.ravel())
```

**Expected outcome:** Reveals whether Hit@K is masking poor precision. MCC is robust to class imbalance and is increasingly expected in pharmacovigilance publications.

---

### REC-5 | Re-test Focal Loss gamma=2.0 with New Dataset

**Files to change:** `scripts3/config.py`  
**Effort:** 30 minutes (config change) + compute time  
**Evidence:** [I2], [I4], [I5] — original Focal Loss paper (Lin et al., 2017) recommends γ=2.0. Current config has `focal_gamma: 1.0` (noted as "best" from old dataset grid search).

**Current state:** `config.py` has `"focal_gamma": 1.0`. The "best: 1.0" was calibrated on the old RXAUI-based dataset, which has now been replaced with the RXCUI-standardised dataset.

**Change:** Add γ=2.0 to the hyperparameter sweep range, or directly test:
```python
"focal_gamma": 2.0,  # re-test: old "best" was on old dataset
```
Run comparison: γ=1.0 vs γ=2.0 on `FAERS_TB` with new standardised dataset.

**Expected outcome:** May recover performance gains masked by the dataset change. Low-cost validation step before any architectural changes.

---

## Tier 2 — Medium Effort (1–3 days each)

---

### REC-6 | Extend Patient Node Features from 7 to 11 Dimensions

**Files to change:** `scripts3/step2_aer_graph_construction.py`, `scripts3/config.py`  
**Effort:** 1 day  
**Evidence:** [I3] achieves AUC=0.902 by including country, comorbidities, and lab values in patient embeddings. [I7] found reporter style (healthcare professional vs. consumer) biases model predictions. [R4] emphasises country/healthcare system as a confounder.

**Current state:** `_patient_feature_vector()` produces only 7 features:
`[age_youth, age_adult, age_elderly, gender_M, gender_F, n_drugs_normed, n_diseases_normed]`

Reporter type (`rpsr_cod`) and outcomes (`outc_cod`) are in `COL` config and tables are loaded, but unused in features.

**Change:** Extend `_patient_feature_vector()` in `step2_aer_graph_construction.py`:
```python
def _patient_feature_vector(row, num_drugs, num_diseases,
                             max_drugs, max_diseases,
                             rpsr_cod="", outc_codes=None) -> np.ndarray:
    # ... existing 7 dims ...
    # New dims:
    country_US      = float(str(row.get(COL["country_code"], "")).upper() == "US")
    reporter_HP     = float(str(rpsr_cod).upper() == "HP")  # healthcare professional
    outcome_serious = float(bool(set(outc_codes or []) & {"DE", "LT", "HO", "DS"}))
    n_outcomes_normed = len(outc_codes or []) / 7.0  # 7 possible outcome codes

    return np.concatenate([
        age_onehot, gender_onehot,
        [nd, ni, country_US, reporter_HP, outcome_serious, n_outcomes_normed]
    ])  # shape (11,)
```
Requires joining report_sources and case_outcomes per patient during `_patient_features()`.

**Expected gain:** +2–5% AUC based on [I3]'s results. Reporter type as a feature directly addresses [I7]'s finding that it confounds predictions. Model input dimension increases 7→11.

---

### REC-7 | Implement Temporal Train-Test Split

**Files to change:** `scripts3/config.py`, `scripts3/step1_preprocessing.py`  
**Effort:** 1 day  
**Evidence:** [I7] is the only paper using temporal split; [R4] calls random CV "optimistic." FAERS PERIOD column (e.g., "21Q3") is directly available in DEMOGRAPHICS.csv.

**Current state:** `step1_preprocessing.py` `split_datasets()` uses random `sklearn.model_selection.train_test_split`. PERIOD column exists but is not used for splitting.

**Change:**

In `config.py`:
```python
SPLIT = {
    "mode"               : "random",    # "random" | "temporal"
    "train"              : 0.75,
    "val"                : 0.125,
    "test"               : 0.125,
    "seed"               : 42,
    "temporal_val_start" : "22Q1",      # cases from this quarter onward → val
    "temporal_test_start": "23Q1",      # cases from this quarter onward → test
}
```

In `step1_preprocessing.py`, add temporal split branch:
```python
if SPLIT.get("mode") == "temporal":
    period_int = demo["PERIOD"].apply(
        lambda p: int(p[:2]) * 10 + int(p[3]) if isinstance(p, str) and len(p) == 4 else 0
    )
    val_start  = int(SPLIT["temporal_val_start"][:2])  * 10 + int(SPLIT["temporal_val_start"][3])
    test_start = int(SPLIT["temporal_test_start"][:2]) * 10 + int(SPLIT["temporal_test_start"][3])
    train_ids = demo[period_int <  val_start]["PRIMARYID"].tolist()
    val_ids   = demo[(period_int >= val_start) & (period_int < test_start)]["PRIMARYID"].tolist()
    test_ids  = demo[period_int >= test_start]["PRIMARYID"].tolist()
```

**Expected gain:** Honest performance estimate. Will likely show AUROC drop vs. random split — this is the real performance. Provides publication-quality methodology and reveals temporal concept drift.

---

### REC-8 | Add Subgroup-Stratified Fairness Evaluation

**Files to change:** `scripts3/step4_training_infonce.py`, `scripts3/step2_aer_graph_construction.py`  
**Effort:** 1 day  
**Evidence:** [R4], [R6] require stratified metrics by age group, sex, country. NFR1: performance gap < 5% between subgroups. None of the 10 implementation papers did this — immediate differentiation for your paper.

**Current state:** `evaluate_all_metrics()` aggregates all patients. No subgroup logic.

**Change:**

In `step2_aer_graph_construction.py`, store patient metadata as separate tensors:
```python
data["patient"].age_group  = torch.tensor(age_group_list,   dtype=torch.long)   # 0=Youth,1=Adult,2=Elderly
data["patient"].gender     = torch.tensor(gender_list,      dtype=torch.long)   # 0=Unknown,1=M,2=F
data["patient"].country_us = torch.tensor(country_us_list,  dtype=torch.float)
```

In `step4_training_infonce.py`, add after standard evaluation:
```python
def subgroup_evaluate(model, graph, device):
    model.eval()
    with torch.no_grad():
        # ... get scores and labels as in evaluate() ...
        pass
    results = {}
    for group_name, group_attr, group_values in [
        ("age",    graph["patient"].age_group,  {0: "Youth", 1: "Adult", 2: "Elderly"}),
        ("gender", graph["patient"].gender,     {1: "Male",  2: "Female"}),
    ]:
        for gv, gname in group_values.items():
            mask_g = (group_attr[eval_mask.cpu()] == gv).numpy()
            if mask_g.sum() < 10:
                continue
            results[f"{group_name}_{gname}"] = compute_auc(
                scores[mask_g], labels[mask_g]
            )
    return results
```

**Expected gain:** Reveals whether the model underperforms for elderly, female, or non-US patients. Required for any thesis chapter on fairness and a novel contribution vs. all 10 implementation papers.

---

### REC-9 | Add GNNExplainer / HGT Attention Weight Extraction

**Files to change:** New file `scripts3/step7_explain.py`, minor change to `scripts3/step3_model_infonce.py`  
**Effort:** 2 days  
**Evidence:** [I3] uses SHAP + attention + counterfactual. [R1, R2] rank SHAP as primary XAI. For GNNs, `torch_geometric.explain.GNNExplainer` is the graph-native approach. NFR5 requires deterministic explanations.

**Current state:** No XAI implemented anywhere in scripts3.

**Change:** Create `scripts3/step7_explain.py`:
```python
"""
step7_explain.py
================
Explainability analysis for trained PreciseADR models.
Implements: (1) GNNExplainer edge/feature importance, (2) HGT attention extraction.
"""
from torch_geometric.explain import Explainer, GNNExplainer

def explain_patient_predictions(model, graph, patient_indices, device):
    """Run GNNExplainer for specified patient node indices."""
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node"),
    )
    results = {}
    for pid_idx in patient_indices:
        explanation = explainer(
            x=graph["patient"].x,
            edge_index=graph[("patient", "takes", "drug")].edge_index,
            index=pid_idx,
        )
        results[pid_idx] = {
            "node_importance": explanation.node_mask.cpu().numpy(),
            "edge_importance": explanation.edge_mask.cpu().numpy(),
        }
    return results
```

Also modify `HGTEncoder.forward()` in `step3_model_infonce.py` to optionally return attention weights:
```python
def forward(self, x_dict, edge_index_dict, return_attention=False):
    attention_weights = {}
    for layer in self.convs:
        x_dict, attn = layer(x_dict, edge_index_dict, return_attention_weights=True)
        if return_attention:
            attention_weights[id(layer)] = attn
    return (x_dict, attention_weights) if return_attention else x_dict
```

**Expected gain:** Enables the "what drove this prediction" narrative required for clinical AI papers. Directly implements [I3]'s SHAP+attention approach. Required for any explainability chapter.

---

### REC-10 | Drug-Removal Counterfactual Explanation

**Files to change:** `scripts3/step7_explain.py` (add to same file as REC-9)  
**Effort:** 1 day  
**Evidence:** [I3] — removing valproic acid from a patient subgraph decreased neutropenia prediction by 36%. Most clinically actionable explanation: "if patient stops drug X, ADR risk decreases by Y%."

**Current state:** No counterfactual logic anywhere.

**Change:** Add to `scripts3/step7_explain.py`:
```python
def drug_removal_counterfactual(model, graph, patient_node_idx, device):
    """
    For a given patient, compute ADR risk change if each drug is removed.
    Returns: list of (drug_name, delta_risk) sorted by impact (descending).
    """
    baseline_scores = _predict_patient(model, graph, patient_node_idx, device)
    results = {}

    takes_ei = graph[("patient", "takes", "drug")].edge_index
    drug_edges = (takes_ei[0] == patient_node_idx).nonzero(as_tuple=True)[0]
    drug_node_indices = takes_ei[1][drug_edges].tolist()

    for drug_idx in drug_node_indices:
        cf_graph = _mask_drug_edges(graph, patient_node_idx, drug_idx)
        cf_scores = _predict_patient(model, cf_graph, patient_node_idx, device)
        delta = (baseline_scores - cf_scores).mean().item()
        drug_name = graph.drug_vocab_inv.get(drug_idx, f"drug_{drug_idx}")
        results[drug_name] = delta

    return sorted(results.items(), key=lambda x: -x[1])
```

**Expected gain:** Answers "which drug is causing this ADR?" — the #1 clinical question. Especially meaningful for TB multi-drug regimens (INH + RIF + PZA + EMB).

---

## Tier 3 — Larger Changes (1–2 weeks each)

---

### REC-11 | Drug Node Features from Disproportionality Statistics

**Files to change:** `scripts3/step2_aer_graph_construction.py`, `scripts3/config.py`  
**Effort:** 2 days  
**Evidence:** [I3] integrates WHO-UMC IC signals as features. Research Gap 4.7 from literature review: "no paper integrates disproportionality statistics directly as node features alongside graph features." This is a novel contribution.

**Current state:** `_drug_features()` returns a one-hot identity matrix. `self.prop` (PROPORTIONATE_ANALYSIS) is loaded and used for signal edges but **not** as drug node features.

**Change:** Replace `_drug_features()` in `step2_aer_graph_construction.py`:
```python
def _drug_features(self) -> torch.Tensor:
    """
    Drug node features: pharmacovigilance statistics from PROPORTIONATE_ANALYSIS.
    Shape: (n_drugs, n_drug + 6) when use_prop_stats is True.
    """
    n_drugs  = len(self.drug_vocab)
    identity = torch.eye(n_drugs, dtype=torch.float)

    if self.prop is None or not CONFIG.get("use_drug_prop_features", False):
        return identity

    drug_col   = self._drug_id_col()
    prop_feats = torch.zeros((n_drugs, 6), dtype=torch.float)
    for drug_key, drug_idx in self.drug_vocab.items():
        sub = self.prop[self.prop[drug_col].astype(str) == str(drug_key)]
        if len(sub) == 0:
            continue
        ror_vals = np.log(sub["ROR"].clip(0.01, 50).astype(float) + 1)
        ic_vals  = sub["IC"].clip(-5, 5).astype(float)
        sig_mask = (sub["IC025"].astype(float) > 0) & (sub["A"].astype(float) >= 3)
        prop_feats[drug_idx] = torch.tensor([
            ror_vals.mean(), ror_vals.max(),
            ic_vals.mean(),  ic_vals.max(),
            sig_mask.sum(),  np.log(len(sub) + 1)
        ])
    return torch.cat([identity, prop_feats], dim=1)
```

Add to `config.py`:
```python
DRUG_FEATURES = {
    "use_prop_stats": False,   # add disproportionality features to drug nodes
    "prop_dim"      : 6,       # [mean_log_ROR, max_log_ROR, mean_IC, max_IC, n_sig, log_reports]
}
```

**Expected gain:** Embeds population-level pharmacovigilance knowledge directly into drug representations. Addresses Research Gap 4.7 as a novel contribution. Expected +3–7% AUC.

---

### REC-12 | Per-ADR Positive Weighting in Focal Loss

**Files to change:** `scripts3/step3_model_infonce.py`, `scripts3/step4_training_infonce.py`  
**Effort:** 1 day  
**Evidence:** [I2] weighted CRF loss; [I5] SMOTE+Tomek Links. Key insight: imbalance varies across ADR types — some ADRs have 2% prevalence, others 40%. Single gamma cannot handle this heterogeneity.

**Current state:** `FocalLoss` uses a single `gamma` uniformly. No `pos_weight` per ADR class.

**Change:**

In `step4_training_infonce.py`, compute per-ADR weights:
```python
y_train   = train_graph["patient"].y.float()
pos_count = y_train.sum(dim=0).clamp(min=1)
neg_count = (y_train.shape[0] - y_train.sum(dim=0)).clamp(min=1)
pos_weight = (neg_count / pos_count).to(device)     # shape (N_adrs,)
```

Modify `FocalLoss` in `step3_model_infonce.py`:
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight  # (N_adrs,) tensor or None

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t   = torch.exp(-bce)
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()
```

**Expected gain:** Better recall on rare ADRs. Directly addresses Research Gap 4.4 (rare ADR prediction gap). Recall@5 and AUPRC should improve for tail ADRs.

---

### REC-13 | Temporal Encoding in Patient Node Features

**Files to change:** `scripts3/step2_aer_graph_construction.py`  
**Effort:** 1 day  
**Evidence:** [I7] temporal split; [R4] concept drift monitoring. FAERS PERIOD column (e.g., "21Q3") contains time information that affects reporting patterns and drug availability.

**Current state:** PERIOD is in `COL["period"]` and demographics table, but not used in patient features.

**Change:** Add `period_normed` as a patient feature dimension:
```python
def _period_to_float(period_str: str) -> float:
    """Convert FAERS PERIOD string (e.g. '21Q3') to normalised year fraction [0, 1]."""
    try:
        year    = 2000 + int(period_str[:2])
        quarter = int(period_str[3])
        return (year + (quarter - 1) * 0.25 - 2015) / 10.0   # 2015=0, 2025=1
    except (ValueError, IndexError):
        return 0.5

# In _patient_features(), add period_normed to feature vector
```

**Expected gain:** Allows the model to learn temporal reporting patterns as a confound, reducing bias from evolving FAERS reporting practices. Enables meaningful temporal generalisation.

---

### REC-14 | ATC-Class One-Hot Encoding for Drug Node Features

**Files to change:** `scripts3/step2_aer_graph_construction.py`, `scripts3/step0_build_datasets.py`  
**Effort:** 2 days  
**Evidence:** [I1] uses ATC hierarchical code one-hot as drug features with strong GCN results. [R2] recommends drug knowledge base integration for KG drug nodes.

**Current state:** Drug features are identity matrices with no pharmacological property encoding. Drugs in the same therapeutic class have no shared representation.

**Change:** Map RXCUI → ATC Level-1 class (16 anatomical main groups) using a pre-built RXCUI→ATC mapping file, then concatenate ATC-L1 one-hot (16-dim) to the drug feature matrix:
```python
ATC_L1_CODES = list("ABCDEFGHJLMNPRSV")   # 16 ATC L1 groups

def _build_drug_atc_features(self, atc_mapping: dict) -> torch.Tensor:
    """
    atc_mapping: {rxcui_str: atc_l1_char}
    Returns: (n_drugs, 16) one-hot ATC-L1 tensor
    """
    n_drugs   = len(self.drug_vocab)
    atc_feats = torch.zeros((n_drugs, len(ATC_L1_CODES)), dtype=torch.float)
    for drug_key, drug_idx in self.drug_vocab.items():
        atc = atc_mapping.get(str(drug_key), "")
        if atc and atc.upper() in ATC_L1_CODES:
            atc_feats[drug_idx, ATC_L1_CODES.index(atc.upper())] = 1.0
    return atc_feats
```

**Expected gain:** Better drug-drug generalisation especially for TB drugs (class J anti-infectives) vs. co-medications (class A alimentary, C cardiovascular for comorbidities). Helps zero-shot generalisation to new drugs.

---

### REC-15 | PubMedBERT Embeddings for ADR and Disease Nodes

**Priority: High**  
**Files to change:** New `scripts3/precompute_node_embeddings.py`, `scripts3/step2_aer_graph_construction.py`, `scripts3/config.py`  
**Effort:** 1 day (pre-compute offline; step2 change is ~10 lines)  
**Evidence:** [R2], [R4] — "disease ontology embeddings" as mandatory node features. [I3] uses "condition and ADR embeddings." [I9] enriches each entity type with intrinsic attribute features. The review explicitly states: *"drug chemical/pharmacological properties"* and *"disease ontology embeddings"* are what node features should capture.

**Current state:** `_adr_features()` and `_disease_features()` at `step2:1007–1009` both return `torch.eye(n, dtype=torch.float)` — a pure identity matrix. This treats every ADR as completely orthogonal. The model has no prior knowledge that *hepatotoxicity* and *liver injury* are related, or that *peripheral neuropathy* and *optic neuritis* share a nerve-damage mechanism.

**Why not ClinicalBERT on MedDRA names?** ClinicalBERT/PubMedBERT both work here. PubMedBERT (`BiomedNLP-BiomedBERT-base`) is preferred because it is pre-trained on PubMed abstracts where MedDRA terminology appears in its natural context. ClinicalBERT is optimised for clinical notes (EHR prose), which is less relevant for MedDRA concept names.

**Step 1 — Pre-compute (run once, ~5 min on CPU):**

Create `scripts3/precompute_node_embeddings.py`:
```python
"""
precompute_node_embeddings.py
==============================
Pre-compute PubMedBERT CLS embeddings for ADR and disease node names.
Run once; outputs saved as .pt tensors and loaded by step2.

Usage:
    cd /home/diyah/PhD_Research/ADR_EHR_Prediction_FAERS
    python scripts3/precompute_node_embeddings.py
"""
import os, torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR      = os.path.join(PROJECT_ROOT, "ADR_Prediction/data/embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model      = AutoModel.from_pretrained(MODEL_NAME).eval()


def encode_terms(terms: list[str]) -> torch.Tensor:
    """CLS-token embedding for a list of short medical terms. Returns (N, 768)."""
    embeddings = []
    with torch.no_grad():
        for term in terms:
            enc = tokenizer(term, return_tensors="pt",
                            truncation=True, max_length=32, padding=True)
            out = model(**enc).last_hidden_state[:, 0, :]   # CLS token
            embeddings.append(out.squeeze(0))
    return torch.stack(embeddings)


if __name__ == "__main__":
    from config import DATA_PATHS, COL

    # ── ADR vocab ────────────────────────────────────────────────────────────
    adr_file = os.path.join(PROJECT_ROOT, DATA_PATHS["adr_reactions"])
    ar       = pd.read_csv(adr_file, dtype=str)
    # Build ordered list of (meddra_code, pt_name) sorted by vocab index
    # Use MEDDRA_CODE as key, PT as display name
    adr_terms = (
        ar[[COL["meddra_code"], COL["adverse_event"]]]
        .drop_duplicates(subset=[COL["meddra_code"]])
        .dropna()
    )
    adr_names  = adr_terms[COL["adverse_event"]].str.lower().tolist()
    adr_codes  = adr_terms[COL["meddra_code"]].tolist()

    print(f"Encoding {len(adr_names)} ADR terms...")
    adr_emb = encode_terms(adr_names)          # (n_adrs, 768)
    torch.save({"codes": adr_codes, "embeddings": adr_emb},
               os.path.join(OUT_DIR, "adr_pubmedbert.pt"))
    print(f"  Saved → {OUT_DIR}/adr_pubmedbert.pt  shape={tuple(adr_emb.shape)}")

    # ── Disease vocab ────────────────────────────────────────────────────────
    dis_file  = os.path.join(PROJECT_ROOT, DATA_PATHS["drug_indications"])
    di        = pd.read_csv(dis_file, dtype=str)
    dis_terms = (
        di[[COL["meddra_code"], COL["indication"]]]
        .drop_duplicates(subset=[COL["meddra_code"]])
        .dropna()
    )
    dis_names = dis_terms[COL["indication"]].str.lower().tolist()
    dis_codes = dis_terms[COL["meddra_code"]].tolist()

    print(f"Encoding {len(dis_names)} disease terms...")
    dis_emb = encode_terms(dis_names)
    torch.save({"codes": dis_codes, "embeddings": dis_emb},
               os.path.join(OUT_DIR, "disease_pubmedbert.pt"))
    print(f"  Saved → {OUT_DIR}/disease_pubmedbert.pt  shape={tuple(dis_emb.shape)}")
```

**Step 2 — Load in step2 (replace one-hot):**

In `step2_aer_graph_construction.py`, replace `_adr_features()` and `_disease_features()`:
```python
_EMB_DIR = os.path.join(PROJECT_ROOT,
                         "ADR_Prediction/data/embeddings")

def _load_bert_embeddings(self, filename: str, vocab: dict) -> torch.Tensor | None:
    """
    Load pre-computed PubMedBERT embeddings, reordered to match vocab index.
    Returns None if file missing or size mismatch → caller falls back to one-hot.
    """
    path = os.path.join(_EMB_DIR, filename)
    if not os.path.exists(path):
        return None
    data     = torch.load(path, weights_only=True)
    codes    = data["codes"]
    emb      = data["embeddings"]            # (N_total, 768)
    code_to_row = {c: i for i, c in enumerate(codes)}
    n        = len(vocab)
    out      = torch.zeros((n, emb.shape[1]), dtype=torch.float)
    matched  = 0
    for key, idx in vocab.items():
        row = code_to_row.get(str(key))
        if row is not None:
            out[idx] = emb[row]
            matched += 1
    print(f"  [BERT feat] {filename}: matched {matched}/{n} vocab entries")
    return out

def _adr_features(self) -> torch.Tensor:
    if NODE_FEATURES.get("use_bert_adr", False):
        emb = self._load_bert_embeddings("adr_pubmedbert.pt", self.adr_vocab)
        if emb is not None:
            return emb
    return torch.eye(len(self.adr_vocab), dtype=torch.float)   # fallback

def _disease_features(self) -> torch.Tensor:
    if NODE_FEATURES.get("use_bert_disease", False):
        emb = self._load_bert_embeddings("disease_pubmedbert.pt", self.disease_vocab)
        if emb is not None:
            return emb
    return torch.eye(len(self.disease_vocab), dtype=torch.float)
```

Add to `config.py`:
```python
NODE_FEATURES = {
    "use_bert_adr"     : False,  # PubMedBERT CLS for ADR nodes (768-dim)
    "use_bert_disease" : False,  # PubMedBERT CLS for disease nodes (768-dim)
    "use_morgan_drug"  : False,  # Morgan fingerprints for drug nodes (see REC-16)
}
```

**Expected gain:** The model enters training with pre-computed semantic similarity between ADRs and diseases. ADRs sharing a MedDRA System Organ Class (SOC) will have similar initial embeddings. Most valuable for rare ADRs (few positive labels) that share structure with well-represented ones. No model architecture changes — `NodeFeatureProjection` already projects any input dim to `hidden_dim`.

---

### REC-16 | Morgan Fingerprint Embeddings for Drug Nodes

**Priority: High**  
**Files to change:** `scripts3/precompute_node_embeddings.py` (extend), `scripts3/step2_aer_graph_construction.py`  
**Effort:** 1 day (SMILES collection is the main work; code is ~20 lines)  
**Evidence:** [I8] Ruseva uses SMILES → molecular descriptors + ATC + fingerprints as drug features. [I10] ToxBERT tokenises SMILES as sequential text for molecular ADR prediction. [R2] recommends drug knowledge base integration. Research Gap 4.6 from literature: SMILES-based structural features have not been combined with population-level graph models.

**Why NOT ClinicalBERT/PubMedBERT on drug names:** Drug names ("isoniazid", "rifampicin") encode almost no pharmacological structure — their text representations primarily reflect clinical co-mention patterns. Two drugs that always appear together in clinical notes (e.g., INH+RIF) will have similar ClinicalBERT embeddings for the wrong reason. Morgan fingerprints encode actual molecular substructure: INH and ethionamide (both thioamides targeting InhA) will be structurally similar; INH and rifampicin will not.

**Current state:** `_drug_features()` at `step2:1001` returns `torch.eye(len(self.drug_vocab))`. All drugs are treated as structurally identical.

**Step 1 — Collect SMILES (one-time manual step):**

For the ~71 TB-relevant drugs in FAERS_TB vocab, SMILES are freely available:
- PubChem: `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/property/IsomericSMILES/JSON`
- DrugBank: already loaded in the pipeline as `COL["drugbank_id"]` — DrugBank XML contains SMILES
- Save as `ADR_Prediction/data/embeddings/rxcui_to_smiles.csv` with columns `RXCUI, SMILES`

**Step 2 — Pre-compute fingerprints (add to precompute_node_embeddings.py):**
```python
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

def compute_morgan_fingerprints(rxcui_smiles: dict,
                                 vocab: dict,
                                 n_bits: int = 512) -> torch.Tensor:
    """
    rxcui_smiles : {rxcui_str: smiles_str}
    vocab        : {rxcui_or_dbid: node_idx}
    Returns      : (n_drugs, n_bits) float tensor; zeros for unknown drugs.
    """
    n_drugs = len(vocab)
    fp_mat  = torch.zeros((n_drugs, n_bits), dtype=torch.float)
    matched = 0
    for drug_key, drug_idx in vocab.items():
        smiles = rxcui_smiles.get(str(drug_key), "")
        mol    = Chem.MolFromSmiles(smiles) if smiles else None
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            fp_mat[drug_idx] = torch.tensor(list(fp), dtype=torch.float)
            matched += 1
    print(f"  Morgan fingerprints: matched {matched}/{n_drugs} drugs")
    return fp_mat

# In __main__:
smiles_df  = pd.read_csv(os.path.join(PROJECT_ROOT,
                 "ADR_Prediction/data/embeddings/rxcui_to_smiles.csv"), dtype=str)
rxcui_smi  = dict(zip(smiles_df["RXCUI"], smiles_df["SMILES"]))
# drug_vocab loaded from the saved graph or rebuilt via step2
drug_fp    = compute_morgan_fingerprints(rxcui_smi, drug_vocab, n_bits=512)
torch.save(drug_fp, os.path.join(OUT_DIR, "drug_morgan512.pt"))
print(f"  Saved → {OUT_DIR}/drug_morgan512.pt  shape={tuple(drug_fp.shape)}")
```

**Step 3 — Load in step2:**
```python
def _drug_features(self) -> torch.Tensor:
    if NODE_FEATURES.get("use_morgan_drug", False):
        path = os.path.join(_EMB_DIR, "drug_morgan512.pt")
        if os.path.exists(path):
            fp_mat = torch.load(path, weights_only=True)   # (n_drugs, 512)
            if fp_mat.shape[0] == len(self.drug_vocab):
                print(f"  [Morgan] Loaded drug fingerprints: {fp_mat.shape}")
                return fp_mat
    return torch.eye(len(self.drug_vocab), dtype=torch.float)   # fallback
```

**Optional — concatenate with disproportionality stats (REC-11):**
```python
# In _drug_features(), after loading fp_mat:
if NODE_FEATURES.get("use_prop_stats", False):
    prop_feats = self._build_drug_prop_features()   # (n_drugs, 6) from REC-11
    return torch.cat([fp_mat, prop_feats], dim=1)   # (n_drugs, 518)
return fp_mat
```

**Expected gain:** Drugs sharing molecular substructure (e.g., all thioamides, all fluoroquinolones) will have similar initial embeddings, enabling the model to generalise across structurally related drugs. Critical for TB pharmacology where INH, ethionamide, and prothionamide target the same enzyme. The `NodeFeatureProjection` handles the dimension change automatically.

---

## Priority Matrix

| # | Recommendation | Files Changed | Effort | Impact | Priority | Literature Basis |
|---|---|---|---|---|---|---|
| **1** | Add AUPRC metric | step4 | 30 min | Measurement | **High** | [R4], [I5] |
| **2** | Add Brier score + ECE | step4 | 2 hrs | Calibration | **High** | [R4], [R6] |
| **3** | Temperature scaling | step4 | 3 hrs | Calibration | **High** | [R4] |
| **4** | Add MCC + macro-F1 | step4 | 1 hr | Measurement | **High** | [I5], [R4] |
| **5** | Re-test focal_gamma=2.0 | config | 30 min | Performance | **High** | [I2], [I4], [I5] |
| **6** | Extend patient features 7→11 dim | step2, config | 1 day | Performance | **High** | [I3], [I7], [R4] |
| **7** | Temporal train-test split | step1, config | 1 day | Methodology | **High** | [I7], [R4] |
| **8** | Subgroup fairness evaluation | step4, step2 | 1 day | Fairness | **High** | [R4], [R6] |
| **9** | GNNExplainer + attention weights | new step7 | 2 days | XAI | **High** | [I3], [R1], [R2] |
| **10** | Drug-removal counterfactuals | step7 | 1 day | XAI | **High** | [I3] |
| **11** | Drug features from prop stats | step2, config | 2 days | Performance | **High** | [I3], [I10], Gap 4.7 |
| **12** | Per-ADR pos_weight in Focal Loss | step3, step4 | 1 day | Performance | Moderate | [I2], [I5], Gap 4.4 |
| **13** | PERIOD temporal patient feature | step2 | 1 day | Performance | Moderate | [I7] |
| **14** | ATC-class drug features | step2, step0 | 2 days | Performance | Moderate | [I1], [R2] |
| **15** | PubMedBERT for ADR + disease nodes | precompute script, step2, config | 1 day | Performance | **High** | [R2], [R4], [I3], [I9] |
| **16** | Morgan fingerprints for drug nodes | precompute script, step2, config | 1 day | Performance | **High** | [I8], [I10], Gap 4.6 |

---

## Recommended Implementation Order for Thesis

```
Phase 1 — Measure (Week 1):
  REC-1 (AUPRC) → REC-2 (Brier/ECE) → REC-4 (F1/MCC) → REC-5 (gamma re-test)

Phase 2 — Features / Performance (Weeks 2–3):
  REC-6 (patient features 11-dim) → REC-11 (drug prop features)
  → Run ablation to confirm gain before continuing

Phase 3 — Methodology Rigour (Week 4):
  REC-3 (temperature scaling) → REC-7 (temporal split) → REC-8 (fairness audit)

Phase 4 — Explainability Chapter (Weeks 5–6):
  REC-9 (GNNExplainer) → REC-10 (counterfactuals)

Phase 5 — Node Embeddings (Weeks 7–8):
  REC-15 (PubMedBERT ADR/disease nodes) → REC-16 (Morgan drug nodes)
  → Collect SMILES for ~71 TB drugs first (prerequisite for REC-16)
  → Run ablation: one-hot vs. BERT-ADR vs. Morgan-drug vs. both combined

Phase 6 — Additional Contributions (as time allows):
  REC-12 (pos_weight) → REC-13 (temporal feature) → REC-14 (ATC features)
```

---

## References

**Review papers:**
- **[R1]** Hauben (2022) — *Do we need explainability?* Pharmacoepidemiol Drug Saf.
- **[R2]** Lee et al. (2023) — *XAI for Patient Safety in Pharmacovigilance.* IEEE Access.
- **[R3]** Algarvio et al. (2025) — *Bayesian Network for Causality Assessment.* Int J Clin Pharm.
- **[R4]** Nwokedi et al. (2025) — *Predictive Models for ADRs using EHRs.* Epidemiol Health Data Insights.
- **[R5]** Malleswari et al. (2026) — *AI to Detect ADRs.* J Advance Future Res.
- **[R6]** CIOMS Working Group XIV (2026) — *AI in Pharmacovigilance.*

**Implementation papers:**
- **[I1]** Kwak et al. (2020) — Drug-Disease Graph GNN. PAKDD 2020.
- **[I2]** Wang et al. (2022) — BERT+wCRF for ADR NER. PLOS Comput Biol.
- **[I3]** Vasanthapuram (2025) — HetGNN + SHAP for ADR prediction. IJAIML.
- **[I4]** Khemani et al. (2025) — GPT/BERT/CNN pharmacovigilance. MethodsX.
- **[I5]** Roberts-Nuttall et al. (2026) — Interpretable RF on drug-target interactions. PLoS One.
- **[I6]** Ferreira-da-Silva et al. (2025) — Causal AI vs. XAI. Int J Clin Pharm.
- **[I7]** Dürlich et al. (2025) — BERT + Integrated Gradients for AER triage. CL4Health @ ACL.
- **[I8]** Ruseva et al. (2025) — MLP on SMILES for ADR prediction. Pharmacia.
- **[I9]** Ni et al. (2026) — KRDQN: KG-reinforced Deep Q-Network. Pharmaceuticals.
- **[I10]** He et al. (2025) — ToxBERT for molecular ADR prediction. J Pharm Analysis.

---

*Generated: 2026-04-13 | scripts3 pipeline | FAERS_TB dataset | standardized_faers_2015_2025*
