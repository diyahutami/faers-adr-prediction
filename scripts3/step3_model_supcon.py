"""
step3_model_v2.py
=================
PreciseADR model architecture - VERSION 2 with improved contrastive learning.

IMPROVEMENTS OVER V1:
  1. Graph-level augmentation (edge dropout) instead of embedding-level dropout
  2. Supervised Contrastive Loss (SupCon) instead of InfoNCE
  3. Separate augmentation dropout (lower) vs encoder dropout (higher)

Components:
  1. NodeFeatureProjection  – projects heterogeneous node features to shared dim d
  2. HGTConvLayers          – L layers of Heterogeneous Graph Transformer (HGT)
  3. GraphAugmentation      – edge dropout for creating augmented graph views
  4. ADRPredictor           – FC layer mapping patient embeddings to ADR probabilities

Training objective:
  L = α · L_SupCon + (1 - α) · L_FocalLoss

This file defines the model classes only; it is imported by step4_training.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Node Feature Projection
# ─────────────────────────────────────────────────────────────────────────────
class NodeFeatureProjection(nn.Module):
    """
    Projects each node type's raw features into a shared embedding dimension d.

    Patient nodes: raw dim = 7 (BOW demographics)
    Drug / Disease / ADR nodes: raw dim = vocab size (one-hot); projected via
        a learnable linear layer.
    """

    def __init__(self, node_types: dict, out_dim: int = 256):
        """
        node_types: dict {node_type_str: in_dim_int}
            e.g. {"patient": 7, "drug": 1200, "disease": 900, "adr": 800}
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
                 dropout: float = 0.5):
        """
        metadata : (node_types, edge_types) as returned by HeteroData.metadata()
        """
        super().__init__()
        self.dropout   = dropout
        self.convs     = nn.ModuleList([
            HGTConv(in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(v.relu(), p=self.dropout,
                                   training=self.training)
                      for k, v in x_dict.items()}
        return x_dict


# ─────────────────────────────────────────────────────────────────────────────
# 3. Graph-Level Augmentation (NEW!)
# ─────────────────────────────────────────────────────────────────────────────
class GraphAugmentation(nn.Module):
    """
    Graph-level augmentation via edge dropout.
    Creates augmented views by randomly dropping edges during message passing.
    
    This is more effective than embedding-level dropout because:
    - Preserves semantic meaning (patients still connected to similar drugs/diseases)
    - Creates structural variation (different message passing paths)
    - Avoids over-corruption (embedding dropout can destroy too much information)
    """
    
    def __init__(self, edge_drop_prob: float = 0.1):
        """
        Args:
            edge_drop_prob: Probability of dropping each edge (default 0.1)
        """
        super().__init__()
        self.edge_drop_prob = edge_drop_prob
    
    def forward(self, edge_index_dict: dict) -> dict:
        """
        Apply edge dropout to create augmented graph.
        
        Args:
            edge_index_dict: Dictionary of edge_index tensors
            
        Returns:
            Augmented edge_index_dict with some edges dropped
        """
        if not self.training:
            # No augmentation during evaluation
            return edge_index_dict
        
        augmented = {}
        for edge_type, edge_index in edge_index_dict.items():
            # Randomly keep edges with probability (1 - edge_drop_prob)
            num_edges = edge_index.shape[1]
            keep_mask = torch.rand(num_edges, device=edge_index.device) > self.edge_drop_prob
            augmented[edge_type] = edge_index[:, keep_mask]
        
        return augmented


# ─────────────────────────────────────────────────────────────────────────────
# 4. Projection Head for Contrastive Learning (NEW!)
# ─────────────────────────────────────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    Maps patient embeddings to a lower-dimensional space where
    contrastive loss is computed.
    
    Following SimCLR and SupCon papers, this helps separate
    representation learning (encoder) from contrastive objective (projection).
    """
    
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64,
                 dropout: float = 0.1):
        """
        Args:
            in_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension  
            out_dim: Output projection dimension
            dropout: Dropout probability (light dropout for regularization)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)  # L2 normalize for cosine similarity


# ─────────────────────────────────────────────────────────────────────────────
# 5. ADR Predictor head
# ─────────────────────────────────────────────────────────────────────────────
class ADRPredictor(nn.Module):
    """FC layer mapping patient embeddings → ADR probability logits."""

    def __init__(self, in_dim: int, n_adrs: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_adrs)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)          # logits; apply sigmoid externally


# ─────────────────────────────────────────────────────────────────────────────
# 6. Loss functions
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Binary Focal Loss for multi-label classification.
    L_focal = -Σ [(1-p_t)^γ · log(p_t)]
    where p_t is the predicted probability of the true class.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # For positive labels: p_t = probs; for negative: p_t = 1 - probs
        p_t   = targets * probs + (1 - targets) * (1 - probs)
        ce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal = ((1 - p_t) ** self.gamma) * ce
        return focal.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    https://arxiv.org/abs/2004.11362
    
    KEY DIFFERENCE FROM InfoNCE:
    - InfoNCE: Only augmented views of the same sample are positive pairs
    - SupCon: Samples with similar labels are also positive pairs
    
    For ADR prediction:
    - Patients with overlapping ADR sets should have similar embeddings
    - More aligned with the downstream task than unsupervised InfoNCE
    
    Formula:
        L_i = -1/|P(i)| Σ_{p∈P(i)} log[ exp(z_i·z_p/τ) / Σ_{a∈A(i)} exp(z_i·z_a/τ) ]
    
    Where:
        P(i) = {p: y_p and y_i have significant overlap} (positive pairs)
        A(i) = all samples except i (anchors)
    """
    
    def __init__(self, temperature: float = 0.07, 
                 similarity_threshold: float = 0.3,
                 use_hard_negatives: bool = False):
        """
        Args:
            temperature: Softmax temperature (default 0.07, typical for SupCon)
            similarity_threshold: Jaccard similarity threshold for positive pairs
                                 (default 0.3 = 30% overlap)
            use_hard_negatives: If True, only use hard negatives (similar embeddings
                               but different labels) for more effective learning
        """
        super().__init__()
        self.tau = temperature
        self.similarity_threshold = similarity_threshold
        self.use_hard_negatives = use_hard_negatives
    
    def compute_label_similarity(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise label similarity using Jaccard index.
        
        Args:
            labels: (N, num_adrs) binary label matrix
            
        Returns:
            (N, N) similarity matrix where sim[i,j] = Jaccard(labels[i], labels[j])
        """
        # Convert to float if needed (bool matmul not supported)
        labels = labels.float()
        
        # Intersection: (N, N) - how many ADRs are shared
        intersection = torch.matmul(labels, labels.T)
        
        # Union: for each pair, total unique ADRs
        labels_sum = labels.sum(dim=1, keepdim=True)  # (N, 1)
        union = labels_sum + labels_sum.T - intersection
        
        # Jaccard similarity
        jaccard = intersection / (union + 1e-8)
        return jaccard
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss with memory-efficient chunking.
        
        Args:
            z1: (N, D) embeddings from view 1 (original graph)
            z2: (N, D) embeddings from view 2 (augmented graph)
            labels: (N, num_adrs) binary label matrix
            
        Returns:
            SupCon loss scalar
        """
        N = z1.shape[0]
        
        # For large N (>1000), use memory-efficient chunked computation
        # This properly includes label-based positives (key difference from InfoNCE)
        if N > 1000:
            # Normalize embeddings
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
            
            # Compute label similarity matrix (needed for positive pairs)
            label_sim = self.compute_label_similarity(labels)  # (N, N)
            pos_label_mask = label_sim >= self.similarity_threshold  # (N, N)
            
            # Concatenate views: [z1; z2] -> (2N, D)
            z_all = torch.cat([z1_norm, z2_norm], dim=0)  # (2N, D)
            
            # Process in chunks to save memory
            chunk_size = 256
            total_loss = 0.0
            
            for i in range(0, N, chunk_size):
                end_i = min(i + chunk_size, N)
                chunk_n = end_i - i
                
                # Get embeddings for this chunk from both views
                z1_chunk = z1_norm[i:end_i]  # (chunk, D)
                z2_chunk = z2_norm[i:end_i]  # (chunk, D)
                
                # Compute similarity with all samples (both views)
                # sim[j, k] = similarity between sample j in chunk and sample k overall
                sim_matrix = torch.matmul(z1_chunk, z_all.T) / self.tau  # (chunk, 2N)
                
                # Build positive mask for this chunk
                pos_mask = torch.zeros(chunk_n, 2*N, dtype=torch.bool, device=z1.device)
                
                # 1. Augmented views are positive: z1[i] ↔ z2[i]
                for j in range(chunk_n):
                    pos_mask[j, N + i + j] = True  # z1[i+j] ↔ z2[i+j]
                
                # 2. Label-based positives from view 1: z1[i] ↔ z1[k] where labels similar
                label_pos_chunk = pos_label_mask[i:end_i, :]  # (chunk, N)
                pos_mask[:, :N] |= label_pos_chunk
                
                # 3. Label-based positives from view 2: z1[i] ↔ z2[k] where labels similar
                pos_mask[:, N:] |= label_pos_chunk
                
                # Remove self-contrast from positives
                for j in range(chunk_n):
                    pos_mask[j, i + j] = False  # z1[i+j] ↔ z1[i+j] excluded
                
                # Compute loss for this chunk
                # Numerator: log-sum-exp over positives
                # Denominator: log-sum-exp over all (excluding self)
                sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
                exp_sim = torch.exp(sim_matrix)
                
                # Mask for all valid anchors (exclude self from view 1)
                anchor_mask = torch.ones(chunk_n, 2*N, dtype=torch.bool, device=z1.device)
                for j in range(chunk_n):
                    anchor_mask[j, i + j] = False  # Exclude self
                
                # Denominator: sum over all anchors
                denom = (exp_sim * anchor_mask).sum(dim=1, keepdim=True)  # (chunk, 1)
                
                # Log-probability for each pair
                log_prob = sim_matrix - torch.log(denom + 1e-8)  # (chunk, 2N)
                
                # Average over positives for each sample
                num_positives = pos_mask.sum(dim=1).clamp(min=1)  # (chunk,)
                loss_per_sample = -(pos_mask * log_prob).sum(dim=1) / num_positives  # (chunk,)
                
                total_loss += loss_per_sample.sum()
            
            return total_loss / N
        
        # Original full implementation for small N (<= 1000)
        # Concatenate both views: [z1; z2] -> (2N, D)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        z = F.normalize(z, dim=-1)
        
        # Compute similarity matrix: (2N, 2N)
        sim_matrix = torch.matmul(z, z.T) / self.tau
        
        # Compute label similarity for positive pair identification
        label_sim = self.compute_label_similarity(labels)  # (N, N)
        
        # Create mask for positive pairs (VECTORIZED for speed)
        pos_mask = torch.zeros(2*N, 2*N, dtype=torch.bool, device=z.device)
        
        # 1. Augmented views are positive pairs (vectorized)
        pos_mask[torch.arange(N, device=z.device), torch.arange(N, 2*N, device=z.device)] = True
        pos_mask[torch.arange(N, 2*N, device=z.device), torch.arange(N, device=z.device)] = True
        
        # 2. Samples with similar labels are positive (vectorized)
        similar_mask = (label_sim >= self.similarity_threshold) & ~torch.eye(N, dtype=torch.bool, device=z.device)
        
        # Expand to 2N×2N by replicating in all 4 quadrants
        pos_mask[:N, :N] |= similar_mask
        pos_mask[:N, N:] |= similar_mask
        pos_mask[N:, :N] |= similar_mask
        pos_mask[N:, N:] |= similar_mask
        
        # Mask out self-contrast (diagonal)
        self_mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
        pos_mask = pos_mask & ~self_mask
        
        # Compute loss
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        exp_sim = torch.exp(sim_matrix)
        neg_mask = ~self_mask
        denom = (exp_sim * neg_mask).sum(dim=1, keepdim=True)
        log_prob = sim_matrix - torch.log(denom + 1e-8)
        num_positives = pos_mask.sum(dim=1).clamp(min=1)
        loss_per_sample = -(pos_mask * log_prob).sum(dim=1) / num_positives
        
        return loss_per_sample.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full PreciseADR model with graph-level augmentation
# ─────────────────────────────────────────────────────────────────────────────
class PreciseADR_v2(nn.Module):
    """
    PreciseADR with improved contrastive learning (VERSION 2).
    
    KEY IMPROVEMENTS:
    1. Graph-level augmentation (edge dropout) creates two views
    2. Projection head for contrastive learning
    3. Original embeddings used for ADR prediction (not augmented)
    
    Forward pass returns:
        logits   – (N_patients, N_ADRs)  ADR prediction logits
        h_orig   – (N_patients, d)       patient embeddings from original graph
        z_orig   – (N_patients, d_proj)  projected embeddings from original graph
        z_aug    – (N_patients, d_proj)  projected embeddings from augmented graph
    """

    def __init__(self,
                 node_dims: dict,
                 metadata: tuple,
                 n_adrs: int,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.5,
                 edge_drop_prob: float = 0.1,
                 projection_dim: int = 64,
                 projection_dropout: float = 0.1):
        """
        node_dims  : {node_type: raw_feature_dim}
        metadata   : HeteroData.metadata() → (node_types, edge_types)
        n_adrs     : number of unique ADR labels
        hidden_dim : embedding dimension for HGT
        num_layers : number of HGT layers
        num_heads  : number of attention heads
        dropout    : dropout for HGT encoder (high for regularization)
        edge_drop_prob : probability of dropping edges for augmentation
        projection_dim : dimension of projection head output
        projection_dropout : dropout for projection head (low)
        """
        super().__init__()
        self.projection = NodeFeatureProjection(node_dims, out_dim=hidden_dim)
        self.encoder    = HGTEncoder(metadata, hidden_dim=hidden_dim,
                                     num_layers=num_layers, num_heads=num_heads,
                                     dropout=dropout)
        self.augmentation = GraphAugmentation(edge_drop_prob=edge_drop_prob)
        self.proj_head    = ProjectionHead(in_dim=hidden_dim, 
                                          hidden_dim=hidden_dim//2,
                                          out_dim=projection_dim,
                                          dropout=projection_dropout)
        self.predictor    = ADRPredictor(hidden_dim, n_adrs)

    def forward(self, x_dict: dict, edge_index_dict: dict,
                patient_mask: torch.Tensor | None = None):
        """
        x_dict          : raw node features per type
        edge_index_dict : edge indices per relation type
        patient_mask    : optional boolean mask to select eval patients
        """
        # Layer 1: project to shared dim
        h = self.projection(x_dict)
        
        # === VIEW 1: Original graph ===
        h1 = self.encoder(h, edge_index_dict)
        h1_patient = h1["patient"]
        if patient_mask is not None:
            h1_patient = h1_patient[patient_mask]
        
        # === VIEW 2: Augmented graph (only during training) ===
        if self.training:
            edge_index_dict_aug = self.augmentation(edge_index_dict)
            # Need fresh forward pass through projection for view 2
            h_proj2 = self.projection(x_dict)
            h2 = self.encoder(h_proj2, edge_index_dict_aug)
            h2_patient = h2["patient"]
            if patient_mask is not None:
                h2_patient = h2_patient[patient_mask]
        else:
            # During eval, no augmentation needed
            h2_patient = h1_patient
        
        # Project both views for contrastive learning
        z1 = self.proj_head(h1_patient)
        z2 = self.proj_head(h2_patient)
        
        # ADR prediction uses original (non-augmented) embeddings
        logits = self.predictor(h1_patient)
        
        return logits, h1_patient, z1, z2

    def predict(self, x_dict: dict, edge_index_dict: dict,
                patient_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return sigmoid probabilities (inference mode)."""
        self.eval()
        with torch.no_grad():
            logits, _, _, _ = self.forward(x_dict, edge_index_dict, patient_mask)
        return torch.sigmoid(logits)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Combined training loss with SupCon
# ─────────────────────────────────────────────────────────────────────────────
class PreciseADRLoss_v2(nn.Module):
    """
    Combined loss function for PreciseADR v2:
        L = α · L_SupCon + (1 - α) · L_FocalLoss
    
    Args:
        alpha: Weight for SupCon loss (default: 0.5)
        tau: Temperature for SupCon (default: 0.07, typical for supervised contrastive)
        gamma: Focal loss gamma parameter (default: 2.0)
        similarity_threshold: Jaccard threshold for positive pairs in SupCon
    """

    def __init__(self, alpha: float = 0.5, tau: float = 0.07, gamma: float = 2.0,
                 similarity_threshold: float = 0.3):
        super().__init__()
        self.alpha    = alpha
        self.focal    = FocalLoss(gamma=gamma)
        self.supcon   = SupConLoss(temperature=tau, 
                                   similarity_threshold=similarity_threshold)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                h_orig: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor):
        """
        Args:
            logits: (N, num_adrs) ADR prediction logits
            targets: (N, num_adrs) ground truth labels
            h_orig: (N, hidden_dim) original patient embeddings (not used, kept for API consistency)
            z1: (N, projection_dim) projected embeddings from view 1
            z2: (N, projection_dim) projected embeddings from view 2
            
        Returns:
            total_loss, focal_loss, supcon_loss
        """
        l_focal  = self.focal(logits, targets)
        l_supcon = self.supcon(z1, z2, targets)
        total    = self.alpha * l_supcon + (1 - self.alpha) * l_focal
        return total, l_focal, l_supcon


# ─────────────────────────────────────────────────────────────────────────────
# 9. Utility: build model from a loaded graph
# ─────────────────────────────────────────────────────────────────────────────
def build_model_from_graph_v2(graph, cfg: dict) -> PreciseADR_v2:
    """
    Convenience factory: instantiate PreciseADR_v2 from a loaded HeteroData graph
    and the MODEL config dict.
    """
    node_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    n_adrs    = len(graph.adr_vocab)
    model     = PreciseADR_v2(
        node_dims          = node_dims,
        metadata           = graph.metadata(),
        n_adrs             = n_adrs,
        hidden_dim         = cfg["embedding_dim"],
        num_layers         = cfg["num_hgt_layers"],
        num_heads          = cfg["num_heads"],
        dropout            = cfg["dropout"],
        edge_drop_prob     = cfg.get("edge_drop_prob", 0.1),
        projection_dim     = cfg.get("projection_dim", 64),
        projection_dropout = cfg.get("projection_dropout", 0.1),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 10. Backward compatibility: aliases for old API
# ─────────────────────────────────────────────────────────────────────────────
# Allow importing the new version as the default
PreciseADR = PreciseADR_v2
PreciseADRLoss = PreciseADRLoss_v2
build_model_from_graph = build_model_from_graph_v2
