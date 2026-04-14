"""
step3_model.py
==============
PreciseADR model architecture (Step 4 of the methodology).

Components:
  1. NodeFeatureProjection  – projects heterogeneous node features to shared dim d
  2. HGTConvLayers          – L layers of Heterogeneous Graph Transformer (HGT)
  3. PatientNodeAugmentation– adds Gaussian noise during training (contrastive view)
  4. ADRPredictor           – FC layer mapping patient embeddings to ADR probabilities

Training objective:
  L = α · L_InfoNCE + (1 - α) · L_FocalLoss

This file defines the model classes only; it is imported by step4_training.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch_geometric.nn import HGTConv, Linear


# ─────────────────────────────────────────────────────────────────────────────
# 1. Node Feature Projection
# ─────────────────────────────────────────────────────────────────────────────
class NodeFeatureProjection(nn.Module):
    """Projects each node type's raw features to a shared embedding dimension d."""

    def __init__(self, node_types: dict, out_dim: int):
        """
        node_types : {node_type_str: in_dim_int}
            e.g. {"patient": 7, "drug": 1200, "disease": 900, "adr": 800}
        out_dim    : shared embedding dimension d
        """
        super().__init__()
        self.projections = nn.ModuleDict({
            nt: Linear(in_dim, out_dim)
            for nt, in_dim in node_types.items()
        })

    def forward(self, x_dict: dict) -> dict:
        return {nt: self.projections[nt](x).relu()
                for nt, x in x_dict.items()
                if nt in self.projections}
    
# ─────────────────────────────────────────────────────────────────────────────
# 2. Heterogeneous Graph Transformer (HGT) layers
# ─────────────────────────────────────────────────────────────────────────────
class HGTEncoder(nn.Module):
    """
    L stacked HGTConv layers operating on a heterogeneous graph.
    Each layer aggregates messages from typed neighbours using
    multi-head attention following Hu et al. (2020).
    """

    def __init__(self, metadata: tuple, hidden_dim: int = 256,
                 num_layers: int = 3, num_heads: int = 8,
                 dropout: float = 0.5, use_checkpoint: bool = False):
        """
        metadata        : (node_types, edge_types) as returned by HeteroData.metadata()
        use_checkpoint  : if True, use gradient checkpointing on each HGT layer.
                          Saves GPU memory at the cost of one extra forward pass per
                          layer during backward. Required for large graphs (FAERS_ALL).
        """
        super().__init__()
        self.dropout        = dropout
        self.use_checkpoint = use_checkpoint
        self.convs          = nn.ModuleList([
            HGTConv(in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads)
            for _ in range(num_layers)
        ])

    def _conv_step(self, conv: nn.Module, x_dict: dict,
                   edge_index_dict: dict) -> dict:
        """Single HGT conv + relu + dropout — extracted for checkpointing."""
        x_dict = conv(x_dict, edge_index_dict)
        return {k: F.dropout(v.relu(), p=self.dropout, training=self.training)
                for k, v in x_dict.items()}

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        for conv in self.convs:
            if self.use_checkpoint and self.training:
                # use_reentrant=False: supports dict inputs/outputs and
                # is the recommended mode in PyTorch ≥ 2.0.
                # Intermediate edge-level attention tensors are NOT stored;
                # they are recomputed during backward, saving 60-80% GPU memory.
                x_dict = grad_checkpoint(
                    self._conv_step, conv, x_dict, edge_index_dict,
                    use_reentrant=False
                )
            else:
                x_dict = self._conv_step(conv, x_dict, edge_index_dict)
        return x_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Patient Node Augmentation (contrastive view)
# ─────────────────────────────────────────────────────────────────────────────
class PatientNodeAugmentation(nn.Module):
    """
    Fully-connected augmentation network that introduces random dropout noise
    to patient representations during training to generate a second view for
    the InfoNCE contrastive objective.

    At inference (eval mode), dropout is disabled → deterministic output.
    """

    def __init__(self, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.fc      = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h_patient: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(h_patient))


# ─────────────────────────────────────────────────────────────────────────────
# 4. ADR Predictor head
# ─────────────────────────────────────────────────────────────────────────────
class ADRPredictor(nn.Module):
    """FC layer mapping patient embeddings → ADR probability logits."""

    def __init__(self, in_dim: int, n_adrs: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_adrs)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)          # logits; apply sigmoid externally


# ─────────────────────────────────────────────────────────────────────────────
# 5. Loss functions
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification with optional per-class
    positive weighting.

    L_focal = -Σ w_c · [(1-p_t)^γ · log(p_t)]

    where p_t is the predicted probability of the true class and w_c is the
    per-ADR class weight (neg_count / pos_count), which up-weights rare ADRs.

    Args:
        gamma      : focusing parameter (paper default 2.0; 0 = standard BCE).
        pos_weight : (n_adrs,) tensor of per-class positive weights, or None.
                     Registered as a buffer so .to(device) moves it automatically.
    """

    def __init__(self, gamma: float = 2.0,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        # register_buffer → moved with .to(device), excluded from state_dict params
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # p_t: probability of the correct class at each position
        p_t   = targets * probs + (1 - targets) * (1 - probs)
        ce    = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,   # None → standard BCE; tensor → weighted
            reduction="none",
        )
        focal = ((1 - p_t) ** self.gamma) * ce
        return focal.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss between the HGT patient representation
    h_patient and the augmented view h_aug.

    Paper formula (Equation 10, Gao et al. 2025):
        L_infonce = -1/B * Σ(i=1 to B) log[ exp(sim(H^L_P[i], H^S_P[i])) / 
                                             Σ(j=1 to B) exp(sim(H^L_P[i], H^S_P[j])) ]
    
    Implementation modes:
    - use_sampled_negatives=False: Exact paper implementation (all negatives)
    - use_sampled_negatives=True: Memory-efficient with sampled negatives (MoCo/SimCLR-style)
                                   Recommended for large datasets and limited GPU memory
    """

    def __init__(self, temperature: float = 0.05, use_sampled_negatives: bool = True,
                 max_negatives: int = 2048):
        super().__init__()
        self.tau = temperature
        self.use_sampled_negatives = use_sampled_negatives
        self.max_negatives = max_negatives

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2 : (N, D) – patient embeddings from two views (H^L_P and H^S_P in paper).
        
        Args:
            z1: Patient embeddings from HGT encoder (N, D)
            z2: Patient embeddings from augmentation layer (N, D)
            
        Returns:
            InfoNCE loss scalar
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        N  = z1.shape[0]
        
        # Decide whether to use sampled negatives based on configuration and dataset size
        use_sampling = self.use_sampled_negatives and (N > self.max_negatives)
        
        if use_sampling:
            # Memory-efficient mode: sample negatives (MoCo/SimCLR-style)
            chunk_size = 256
        else:
            # Exact paper implementation: use all negatives
            chunk_size = min(512, N)
            
        total_loss = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        
        for i in range(0, N, chunk_size):
            end_i = min(i + chunk_size, N)
            z1_chunk = z1[i:end_i]  # (chunk, D)
            z2_chunk = z2[i:end_i]  # (chunk, D)
            chunk_n = end_i - i
            
            # Compute similarity for this chunk
            # Positive pairs: z1_chunk with z2_chunk (same indices)
            pos_sim = torch.sum(z1_chunk * z2_chunk, dim=-1) / self.tau  # (chunk,)
            
            # For negative samples, use a subset to save memory
            if use_sampling:
                # MEMORY-EFFICIENT MODE: Sample negative indices (excluding current chunk indices)
                all_indices = torch.arange(N, device=z1.device)
                chunk_indices = torch.arange(i, end_i, device=z1.device)
                
                # Create mask for sampling negatives
                mask = torch.ones(N, dtype=torch.bool, device=z1.device)
                mask[chunk_indices] = False
                neg_indices = all_indices[mask]
                
                # Sample a subset of negatives
                n_negs = min(self.max_negatives, len(neg_indices))
                sampled_neg_idx = neg_indices[torch.randperm(len(neg_indices), device=z1.device)[:n_negs]]
                
                # Compute negative similarities
                z2_negs = z2[sampled_neg_idx]  # (n_negs, D)
                neg_sim = torch.matmul(z1_chunk, z2_negs.T) / self.tau  # (chunk, n_negs)
            else:
                # EXACT PAPER FORMULA: Use all samples as negatives
                # Use all samples as negatives
                neg_sim = torch.matmul(z1_chunk, z2.T) / self.tau  # (chunk, N)
                
                # Remove positive pairs from negatives
                mask = torch.zeros(chunk_n, N, device=z1.device, dtype=torch.bool)
                for j in range(chunk_n):
                    mask[j, i + j] = True
                neg_sim = neg_sim.masked_fill(mask, -1e9)
            
            # InfoNCE loss for this chunk
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (chunk, 1+n_negs)
            labels = torch.zeros(chunk_n, dtype=torch.long, device=z1.device)
            
            chunk_loss = F.cross_entropy(logits, labels, reduction='sum')
            total_loss = total_loss + chunk_loss
        
        return total_loss / N


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full PreciseADR model
# ─────────────────────────────────────────────────────────────────────────────
class PreciseADR(nn.Module):
    """
    Full PreciseADR model combining all four layers.

    Forward pass returns:
        logits   – (N_patients, N_ADRs)  ADR prediction logits
        h_orig   – (N_patients, d)       patient embeddings from HGT
        h_aug    – (N_patients, d)       augmented patient embeddings
    """

    def __init__(self,
                 node_dims: dict,
                 metadata: tuple,
                 n_adrs: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.5,
                 use_checkpoint: bool = False):
        """
        node_dims      : {node_type: raw_feature_dim}
        metadata       : HeteroData.metadata() → (node_types, edge_types)
        n_adrs         : number of unique ADR labels
        use_checkpoint : enable gradient checkpointing in HGTEncoder (reduces
                         GPU memory for large graphs at cost of extra compute)
        """
        super().__init__()
        self.projection = NodeFeatureProjection(node_dims, out_dim=hidden_dim)
        self.encoder    = HGTEncoder(metadata, hidden_dim=hidden_dim,
                                     num_layers=num_layers, num_heads=num_heads,
                                     dropout=dropout,
                                     use_checkpoint=use_checkpoint)
        self.augment    = PatientNodeAugmentation(dim=hidden_dim, dropout=dropout)
        self.predictor  = ADRPredictor(hidden_dim, n_adrs)

    def forward(self, x_dict: dict, edge_index_dict: dict,
                patient_mask: None):
        """
        x_dict          : raw node features per type
        edge_index_dict : edge indices per relation type
        patient_mask    : optional boolean mask to select eval patients
        """
        # Layer 1: project to shared dim
        h = self.projection(x_dict)

        # Layer 2: HGT message passing
        h = self.encoder(h, edge_index_dict)

        h_patient = h["patient"]
        if patient_mask is not None:
            h_patient = h_patient[patient_mask]

        # Layer 3: augmented view (dropout active during training)
        h_aug = self.augment(h_patient)

        # Layer 4: predict ADRs
        logits = self.predictor(h_aug)   # or h_patient at inference

        return logits, h_patient, h_aug

    def embed_patients(self, x_dict: dict, edge_index_dict: dict,
                       patient_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Run projection + HGT encoder only → return patient node embeddings (N, D).

        Used in two-pass training: pass 1 runs with torch.no_grad() to get
        h_patient cheaply (no activation storage); pass 2 runs with grad tracking
        to backprop through the GNN using the accumulated h_patient gradient.
        """
        h = self.projection(x_dict)
        h = self.encoder(h, edge_index_dict)
        h_patient = h["patient"]
        if patient_mask is not None:
            h_patient = h_patient[patient_mask]
        return h_patient

    def predict(self, x_dict: dict, edge_index_dict: dict,
                patient_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return sigmoid probabilities (inference mode)."""
        self.eval()
        with torch.no_grad():
            logits, _, _ = self.forward(x_dict, edge_index_dict, patient_mask)
        return torch.sigmoid(logits)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Combined training loss
# ─────────────────────────────────────────────────────────────────────────────
class PreciseADRLoss(nn.Module):
    """
    Combined loss function for PreciseADR (per paper Equation 11):
        L = α · L_InfoNCE + (1 - α) · L_FocalLoss
    
    Args:
        alpha: Weight for InfoNCE loss (paper default: 0.5)
        tau: Temperature for InfoNCE (paper default: 0.05)
        gamma: Focal loss gamma parameter (paper default: 2.0)
        use_sampled_negatives: Use memory-efficient sampled negatives in InfoNCE
        max_negatives: Maximum number of negative samples when sampling
    """

    def __init__(self, alpha: float = 0.5, tau: float = 0.05, gamma: float = 2.0,
                 pos_weight: torch.Tensor = None,
                 use_sampled_negatives: bool = True, max_negatives: int = 2048):
        super().__init__()
        self.alpha    = alpha
        self.focal    = FocalLoss(gamma=gamma, pos_weight=pos_weight)
        self.infonce  = InfoNCELoss(temperature=tau,
                                    use_sampled_negatives=use_sampled_negatives,
                                    max_negatives=max_negatives)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                h_orig: torch.Tensor, h_aug: torch.Tensor):
        l_focal   = self.focal(logits, targets)
        l_infonce = self.infonce(h_orig, h_aug)
        return self.alpha * l_infonce + (1 - self.alpha) * l_focal, l_focal, l_infonce


# ─────────────────────────────────────────────────────────────────────────────
# Utility: build model from a loaded graph
# ─────────────────────────────────────────────────────────────────────────────
def build_model_from_graph(graph, cfg: dict) -> PreciseADR:
    """
    Convenience factory: instantiate PreciseADR from a loaded HeteroData graph
    and the MODEL config dict.
    """
    node_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    n_adrs    = len(graph.adr_vocab)
    model     = PreciseADR(
        node_dims      = node_dims,
        metadata       = graph.metadata(),
        n_adrs         = n_adrs,
        hidden_dim     = cfg["embedding_dim"],
        num_layers     = cfg["num_hgt_layers"],
        num_heads      = cfg["num_heads"],
        dropout        = cfg["dropout"],
        use_checkpoint = cfg.get("use_gradient_checkpointing", False),
    )
    return model
