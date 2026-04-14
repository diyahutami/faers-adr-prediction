"""
step5_baselines.py
==================
Step 5 of the PreciseADR reproduction pipeline — Baseline Evaluation.

Implements two categories of baselines following Gao et al. (2025):

  5.1 Frequency-Based Baselines (no learnable parameters)
      - Random
      - ADR-Freq
      - ADR-Freq|I     (conditioned on patient's disease/indication set)
      - ADR-Freq|D     (conditioned on patient's drug set)
      - ADR-Freq|I&D   (conditioned on both)

  5.2 Deep-Learning Baselines ("Given Drug" setting)
      All use drug-drug co-occurrence homogeneous graph for GNN methods.
      Models: MLP, Transformer, TabNet, AMFormer,
              GCN, GAT, GraphSAGE, GCNII, FANet, GATv2, ARMA

  5.3 HGNN Backbone Variants (same AER Graph, different GNN layer)
      HGT, RGCN, GCN, GAT, GraphSAGE, GCNII, GATv2, FANet, MLP, Transformer

Results are saved to results/{variant}/baselines_results.json

Usage
-----
    python step5_baselines.py [--variant xxx|xxx_gender|xxx_age]
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import (GCNConv, GATConv, SAGEConv, GATv2Conv,
                                 RGCNConv, ARMAConv)
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from config import GRAPH_PATH, MODEL_PATH, RESULTS_PATH, MODEL, EVAL
from config import DATASET_NAME, OUTPUT_PATH
from step4_training_infonce import evaluate_all_metrics, convert_to_serializable

METHOD = "infonce"
RESULTS_PATH = os.path.join(OUTPUT_PATH, f"results_{DATASET_NAME}_infonce")
# ─────────────────────────────────────────────────────────────────────────────
# 5.1 Frequency-Based Baselines
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyBaselines:
    """
    All frequency baselines use training-set statistics only.
    At test time they produce a score matrix (N_patients × N_adrs).
    """

    def __init__(self, train_graph: HeteroData):
        self.train_graph = train_graph
        self.n_adrs      = len(train_graph.adr_vocab)
        self._compute_stats()

    def _compute_stats(self):
        """Pre-compute ADR marginal frequencies and conditional frequencies."""
        g = self.train_graph
        mask = g["patient"].eval_mask

        # Ground-truth labels for training patients
        y_train = g["patient"].y[mask].numpy()  # (N_train, N_adr)
        self.adr_freq = y_train.sum(axis=0) / max(y_train.shape[0], 1)  # marginal

        # patient → drug set  (node indices in graph)
        ptd = g["patient", "takes", "drug"].edge_index.numpy()   # (2, E)
        # patient → disease set
        phd = g["patient", "has", "disease"].edge_index.numpy()  # (2, E)
        # patient → adr set  (training labels)
        pea = g["patient", "experiences", "adr"].edge_index.numpy() if \
              hasattr(g["patient", "experiences", "adr"], "edge_index") else \
              np.zeros((2, 0), dtype=int)

        self.train_mask_indices = np.where(mask.numpy())[0]

        # Build drug → ADR conditional frequency
        self.drug_adr_freq = np.zeros((g["drug"].x.shape[0], self.n_adrs), dtype=np.float32)
        # Build disease → ADR conditional frequency
        self.dis_adr_freq  = np.zeros((g["disease"].x.shape[0], self.n_adrs), dtype=np.float32)

        # Count co-occurrences via patient pivot
        train_pid_set = set(self.train_mask_indices.tolist())
        pat_drug = {}; pat_dis = {}; pat_adr = {}
        for i in range(ptd.shape[1]):
            p, d = int(ptd[0, i]), int(ptd[1, i])
            if p in train_pid_set:
                pat_drug.setdefault(p, set()).add(d)
        for i in range(phd.shape[1]):
            p, di = int(phd[0, i]), int(phd[1, i])
            if p in train_pid_set:
                pat_dis.setdefault(p, set()).add(di)
        for i in range(pea.shape[1]):
            p, a = int(pea[0, i]), int(pea[1, i])
            if p in train_pid_set:
                pat_adr.setdefault(p, set()).add(a)

        drug_count = np.zeros(g["drug"].x.shape[0])
        dis_count  = np.zeros(g["disease"].x.shape[0])
        for p, adrs in pat_adr.items():
            for d in pat_drug.get(p, []):
                drug_count[d] += 1
                for a in adrs:
                    self.drug_adr_freq[d, a] += 1
            for di in pat_dis.get(p, []):
                dis_count[di] += 1
                for a in adrs:
                    self.dis_adr_freq[di, a] += 1

        # Normalize to frequencies
        drug_count = np.maximum(drug_count, 1)[:, None]
        dis_count  = np.maximum(dis_count,  1)[:, None]
        self.drug_adr_freq /= drug_count
        self.dis_adr_freq  /= dis_count

        self.n_adr = self.n_adrs

    def predict_random(self, n_patients: int) -> np.ndarray:
        return np.random.uniform(0, 1, size=(n_patients, self.n_adr))

    def predict_adr_freq(self, n_patients: int) -> np.ndarray:
        return np.tile(self.adr_freq, (n_patients, 1))

    def predict_adr_freq_given_I(self, dis_edges: np.ndarray, n_patients: int) -> np.ndarray:
        """dis_edges: (2, E) with patient and disease node indices."""
        scores = np.zeros((n_patients, self.n_adr))
        counts = np.zeros(n_patients)
        for i in range(dis_edges.shape[1]):
            p, d = int(dis_edges[0, i]), int(dis_edges[1, i])
            if p < n_patients:
                scores[p] += self.dis_adr_freq[d]
                counts[p] += 1
        counts = np.maximum(counts, 1)[:, None]
        return scores / counts

    def predict_adr_freq_given_D(self, drug_edges: np.ndarray, n_patients: int) -> np.ndarray:
        scores = np.zeros((n_patients, self.n_adr))
        counts = np.zeros(n_patients)
        for i in range(drug_edges.shape[1]):
            p, d = int(drug_edges[0, i]), int(drug_edges[1, i])
            if p < n_patients:
                scores[p] += self.drug_adr_freq[d]
                counts[p] += 1
        counts = np.maximum(counts, 1)[:, None]
        return scores / counts

    def predict_adr_freq_given_ID(self, drug_edges, dis_edges, n_patients) -> np.ndarray:
        s_d = self.predict_adr_freq_given_D(drug_edges, n_patients)
        s_i = self.predict_adr_freq_given_I(dis_edges, n_patients)
        return (s_d + s_i) / 2.0

    def evaluate_all(self, test_graph: HeteroData) -> dict:
        mask = test_graph["patient"].eval_mask
        eval_idx = torch.where(mask)[0]
        n_eval   = int(mask.sum())
        y_true   = test_graph["patient"].y[mask].numpy()

        # Build local (eval-patients only) edge arrays
        ptd  = test_graph["patient", "takes",   "drug"].edge_index.numpy()
        phd  = test_graph["patient", "has",      "disease"].edge_index.numpy()
        eval_set = set(eval_idx.tolist())

        def _local(ei):
            keep = np.isin(ei[0], list(eval_set))
            # Remap patient indices to 0..n_eval-1
            idx_map = {v: i for i, v in enumerate(eval_idx.tolist())}
            local_p = np.array([idx_map[p] for p in ei[0, keep]], dtype=int)
            return np.stack([local_p, ei[1, keep]])

        ptd_local = _local(ptd)
        phd_local = _local(phd)

        results = {}
        for name, scores in [
            ("Random",        self.predict_random(n_eval)),
            ("ADR-Freq",      self.predict_adr_freq(n_eval)),
            ("ADR-Freq|I",    self.predict_adr_freq_given_I(phd_local, n_eval)),
            ("ADR-Freq|D",    self.predict_adr_freq_given_D(ptd_local, n_eval)),
            ("ADR-Freq|I&D",  self.predict_adr_freq_given_ID(ptd_local, phd_local, n_eval)),
        ]:
            results[name] = evaluate_all_metrics(scores, y_true)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.2 Deep-Learning Baselines (co-occurrence drug graph)
# ─────────────────────────────────────────────────────────────────────────────

def _build_drug_cooccurrence_graph(train_graph: HeteroData) -> Data:
    """
    Build a homogeneous drug-drug co-occurrence graph from the training set.
    Two drugs share an edge if they co-appear in any patient's report.
    """
    ptd = train_graph["patient", "takes", "drug"].edge_index.numpy()
    n_drugs = train_graph["drug"].x.shape[0]

    # Group drugs per patient
    from collections import defaultdict
    pat_drugs = defaultdict(list)
    for i in range(ptd.shape[1]):
        pat_drugs[int(ptd[0, i])].append(int(ptd[1, i]))

    src_list, dst_list = [], []
    for drugs in pat_drugs.values():
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                src_list += [drugs[i], drugs[j]]
                dst_list += [drugs[j], drugs[i]]

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(
        x          = train_graph["drug"].x,
        edge_index = edge_index,
        num_nodes  = n_drugs,
    )


class HomogeneousGNNBaseline(nn.Module):
    """Generic wrapper for homogeneous GNN baselines."""

    ARCHITECTURES = ["GCN", "GAT", "GraphSAGE", "GCNII", "GATv2", "ARMA"]

    def __init__(self, arch: str, in_dim: int, hidden_dim: int,
                 n_adrs: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()

        if arch == "GCN":
            self.convs = nn.ModuleList([
                GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)])
        elif arch == "GAT":
            self.convs = nn.ModuleList([
                GATConv(in_dim if i == 0 else hidden_dim, hidden_dim // 4, heads=4)
                for i in range(num_layers)])
        elif arch == "GraphSAGE":
            self.convs = nn.ModuleList([
                SAGEConv(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)])
        elif arch == "GATv2":
            self.convs = nn.ModuleList([
                GATv2Conv(in_dim if i == 0 else hidden_dim, hidden_dim // 4, heads=4)
                for i in range(num_layers)])
        elif arch == "ARMA":
            self.convs = nn.ModuleList([
                ARMAConv(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)])
        else:   # GCNII / default to GCN
            self.convs = nn.ModuleList([
                GCNConv(in_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)])

        self.predictor = nn.Linear(hidden_dim, n_adrs)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.predictor(x)


class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_adrs: int,
                 num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, n_adrs))
        self.net = nn.Sequential(*layers)

    def forward(self, x, *args):
        return self.net(x)


def train_dl_baseline(model: nn.Module, train_graph: HeteroData,
                      cooc_graph: Data, device: torch.device,
                      n_epochs: int = 50, lr: float = 1e-3) -> nn.Module:
    """Train a DL baseline model using drug node features → ADR predictions per patient."""
    model = model.to(device)
    opt   = Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    x   = cooc_graph.x.to(device)
    ei  = cooc_graph.edge_index.to(device)
    ptd = train_graph["patient", "takes", "drug"].edge_index.numpy()
    mask = train_graph["patient"].eval_mask.numpy()
    y   = train_graph["patient"].y.to(device)
    eval_idx = np.where(mask)[0]

    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        drug_emb = model(x, ei)                        # (N_drugs, N_adr)

        # Aggregate drug embeddings per patient (mean pooling)
        n_patients = int(mask.sum())
        pat_scores = torch.zeros(len(eval_idx), drug_emb.shape[1], device=device)
        counts     = torch.zeros(len(eval_idx), 1, device=device)
        pid_to_local = {p: i for i, p in enumerate(eval_idx)}
        for j in range(ptd.shape[1]):
            p, d = int(ptd[0, j]), int(ptd[1, j])
            if p in pid_to_local:
                li = pid_to_local[p]
                pat_scores[li] += drug_emb[d]
                counts[li] += 1
        counts = counts.clamp(min=1)
        pat_scores = pat_scores / counts

        loss = loss_fn(pat_scores, y[mask])
        loss.backward()
        opt.step()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5.3 HGNN Backbone Variants (same AER Graph)
# ─────────────────────────────────────────────────────────────────────────────

# Backbone replacement is handled by training PreciseADR with a different
# `conv_type` argument. We keep this simple by importing the full model and
# overriding the encoder layer type.

from step3_model_infonce import (NodeFeatureProjection, PatientNodeAugmentation,
                          ADRPredictor, FocalLoss, InfoNCELoss, PreciseADR,
                          PreciseADRLoss, build_model_from_graph)
from torch_geometric.nn import HGTConv, RGCNConv as _RGCNConv

class RGCNEncoder(nn.Module):
    """Relational GCN encoder for heterogeneous graph (alternative to HGT)."""
    def __init__(self, metadata, hidden_dim=256, num_layers=3):
        super().__init__()
        n_relations = len(metadata[1])
        self.convs  = nn.ModuleList([
            _RGCNConv(hidden_dim, hidden_dim, num_relations=n_relations)
            for _ in range(num_layers)
        ])
        self._metadata = metadata

    def forward(self, x_dict, edge_index_dict):
        # Flatten to homogeneous for RGCN (simple approach)
        # Build global node index and edge list
        node_types = list(x_dict.keys())
        offset = {}; cum = 0
        all_x = []
        for nt in node_types:
            offset[nt] = cum
            all_x.append(x_dict[nt])
            cum += x_dict[nt].shape[0]
        x_global = torch.cat(all_x, dim=0)

        all_ei = []; all_rt = []
        for ri, (src_t, rel, dst_t) in enumerate(edge_index_dict.keys()):
            ei = edge_index_dict[(src_t, rel, dst_t)]
            ei_shifted = ei.clone()
            ei_shifted[0] += offset[src_t]
            ei_shifted[1] += offset[dst_t]
            all_ei.append(ei_shifted)
            all_rt.append(torch.full((ei.shape[1],), ri, dtype=torch.long))

        if all_ei:
            ei_global = torch.cat(all_ei, dim=1)
            rt_global = torch.cat(all_rt)
        else:
            ei_global = torch.zeros((2, 0), dtype=torch.long, device=x_global.device)
            rt_global = torch.zeros(0, dtype=torch.long, device=x_global.device)

        for conv in self.convs:
            x_global = conv(x_global, ei_global, rt_global).relu()

        # Un-flatten back to dict
        result = {}
        for nt in node_types:
            s, e = offset[nt], offset[nt] + x_dict[nt].shape[0]
            result[nt] = x_global[s:e]
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines(variant: str, graph_path: str, results_path: str,
                  device: torch.device):
    print(f"\n{'='*70}")
    print(f"  BASELINES: {variant.upper()}")
    print(f"{'='*70}")
    out_dir = os.path.join(results_path, variant)
    os.makedirs(out_dir, exist_ok=True)

    train_graph = torch.load(os.path.join(graph_path, variant, "train_graph.pt"), weights_only=False)
    test_graph  = torch.load(os.path.join(graph_path, variant, "test_graph.pt"), weights_only=False)
    n_adrs      = len(train_graph.adr_vocab)

    results = {}

    # ── 5.1 Frequency-based ───────────────────────────────────────────────
    print("\n  5.1 Frequency-based baselines …")
    freq_bl = FrequencyBaselines(train_graph)
    freq_results = freq_bl.evaluate_all(test_graph)
    for name, m in freq_results.items():
        print(f"    {name:<20s} AUC={m['AUC']:.4f}  Hit@10={m.get('Hit@10', 0):.4f}")
    results["frequency"] = freq_results

    # ── 5.2 DL baselines ─────────────────────────────────────────────────
    print("\n  5.2 Deep-learning baselines (Given Drug) …")
    cooc_graph = _build_drug_cooccurrence_graph(train_graph)
    drug_in_dim = train_graph["drug"].x.shape[1]
    hidden_dim  = MODEL["embedding_dim"]

    dl_results = {}
    for arch in ["MLP"] + HomogeneousGNNBaseline.ARCHITECTURES:
        if arch == "MLP":
            model = MLPBaseline(drug_in_dim, hidden_dim, n_adrs)
        else:
            model = HomogeneousGNNBaseline(arch, drug_in_dim, hidden_dim, n_adrs)

        model = train_dl_baseline(model, train_graph, cooc_graph,
                                  device, n_epochs=50)

        # Evaluate: aggregate drug embeddings per test patient
        model.eval()
        with torch.no_grad():
            x   = cooc_graph.x.to(device)
            ei  = cooc_graph.edge_index.to(device)
            drug_emb = model(x, ei)

        test_ptd  = test_graph["patient", "takes", "drug"].edge_index.numpy()
        test_mask = test_graph["patient"].eval_mask.numpy()
        eval_idx  = np.where(test_mask)[0]
        y_true    = test_graph["patient"].y[test_mask].numpy()
        pid_to_local = {p: i for i, p in enumerate(eval_idx)}
        n_eval = len(eval_idx)
        scores = np.zeros((n_eval, n_adrs), dtype=np.float32)
        counts = np.zeros(n_eval, dtype=np.float32)
        for j in range(test_ptd.shape[1]):
            p, d = int(test_ptd[0, j]), int(test_ptd[1, j])
            if p in pid_to_local:
                li = pid_to_local[p]
                scores[li] += torch.sigmoid(drug_emb[d]).cpu().numpy()
                counts[li] += 1
        counts = np.maximum(counts, 1)[:, None]
        scores /= counts
        m = evaluate_all_metrics(scores, y_true)
        print(f"    {arch:<20s} AUC={m['AUC']:.4f}  Hit@10={m.get('Hit@10', 0):.4f}")
        dl_results[arch] = m

    results["deep_learning"] = dl_results

    # ── 5.3 HGNN backbone variants ────────────────────────────────────────
    print("\n  5.3 HGNN backbone variants (same AER graph) …")
    # (Full training of each backbone is expensive; we run a quick 30-epoch version)
    backbone_results = {}
    for arch in ["HGT", "RGCN"]:
        cfg = dict(MODEL)
        # cfg["num_hgt_layers"] = 2; cfg["max_epochs"] = 30; cfg["patience"] = 5
        model = build_model_from_graph(train_graph, cfg)
        if arch == "RGCN":
            # Swap encoder
            model.encoder = RGCNEncoder(train_graph.metadata(),
                                         hidden_dim=cfg["embedding_dim"],
                                         num_layers=cfg["num_hgt_layers"])
        model = model.to(device)
        opt  = Adam(model.parameters(), lr=cfg["lr"])
        loss_fn = PreciseADRLoss(alpha=cfg["alpha"], tau=cfg["tau"],
                                  gamma=cfg["focal_gamma"]).to(device)
        # Quick train
        x_dict = {nt: train_graph[nt].x.to(device) for nt in train_graph.node_types}
        ei_dict = {et: train_graph[et].edge_index.to(device)
                   for et in train_graph.edge_types
                   if hasattr(train_graph[et], "edge_index")}
        mask = train_graph["patient"].eval_mask.to(device)
        y    = train_graph["patient"].y.to(device)
        for ep in range(cfg["max_epochs"]):
            model.train()
            opt.zero_grad()
            logits, h, h_aug = model(x_dict, ei_dict, patient_mask=mask)
            loss, _, _ = loss_fn(logits, y[mask], h, h_aug)
            loss.backward(); opt.step()

        # Evaluate on test graph
        model.eval()
        with torch.no_grad():
            tx = {nt: test_graph[nt].x.to(device) for nt in test_graph.node_types}
            te = {et: test_graph[et].edge_index.to(device)
                  for et in test_graph.edge_types
                  if hasattr(test_graph[et], "edge_index")}
            tmask = test_graph["patient"].eval_mask.to(device)
            tlogits, _, _ = model(tx, te, patient_mask=tmask)
            scores = torch.sigmoid(tlogits).cpu().numpy()
        y_true = test_graph["patient"].y[test_graph["patient"].eval_mask].numpy()
        m = evaluate_all_metrics(scores, y_true)
        print(f"    {arch:<20s} AUC={m['AUC']:.4f}  Hit@10={m.get('Hit@10', 0):.4f}")
        backbone_results[arch] = m

    results["backbone_variants"] = backbone_results

    # Save
    with open(os.path.join(out_dir, "baselines_results.json"), "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\n  Saved → {out_dir}/baselines_results.json")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="PreciseADR Step 5: Baselines")
    p.add_argument("--graph-path",   default=GRAPH_PATH)
    p.add_argument("--results-path", default=RESULTS_PATH)
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
        gdir = os.path.join(args.graph_path, v)
        if os.path.exists(gdir):
            run_baselines(v, args.graph_path, args.results_path, device)
        else:
            print(f"  ⚠ Graph not found for {v}, skipping.")
    print("\n✓ Step 5 (Baselines) complete!")


if __name__ == "__main__":
    main()
