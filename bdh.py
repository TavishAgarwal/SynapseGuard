"""
bdh.py — Biological Dense-Sparse Hybrid (BDH) Toy Baseline Implementation (~10M Parameters)

DISCLAIMER & CLAIM BOUNDARY [ESTABLISHED / MEASURED]:
This is a ~10M parameter educational/baseline instrumentation model designed to demonstrate
the core BDH hypothesis (input predictability vs activation sparsity) on local CPU/MPS hardware.
It is NOT a claim about frontier BDH capabilities or multi-billion parameter scaled performance.

Architecture:
- Input Token Embedding (vocab_size=1000, d_model=128)
- Sparse Monosemantic Expansion Layer (d_model=128 -> d_sparse=2048 with top-k / thresholded k-sparsity)
- Output Decoder Layer (d_sparse=2048 -> vocab_size=1000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseMonosemanticLayer(nn.Module):
    """
    Biologically-inspired sparse expansion layer.
    Maps dense hidden representations to high-dimensional sparse latent representations.
    """
    def __init__(self, d_model: int = 128, d_sparse: int = 2048, k_sparsity: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_sparse = d_sparse
        self.k_sparsity = k_sparsity
        
        # Expansion weights
        self.W_gate = nn.Linear(d_model, d_sparse, bias=True)
        self.W_val = nn.Linear(d_model, d_sparse, bias=False)
        
    def forward(self, x: torch.Tensor, threshold_mode: bool = True):
        """
        x: [batch_size, seq_len, d_model]
        Returns:
            latents: [batch_size, seq_len, d_sparse]
            sparsity_metrics: dict containing L0 norm and active ratio
        """
        raw_gate = F.relu(self.W_gate(x))
        values = self.W_val(x)
        
        if threshold_mode:
            # Top-k activation threshold per token
            topk_vals, topk_indices = torch.topk(raw_gate, k=self.k_sparsity, dim=-1)
            min_topk = topk_vals[..., -1:]
            select = (raw_gate >= min_topk).float()
            sparse_latents = raw_gate * select * values
        else:
            sparse_latents = raw_gate * values
            
        # Compute activation sparsity metrics
        active_mask = (torch.abs(sparse_latents) > 1e-5).float()
        l0_norm = torch.sum(active_mask, dim=-1) # Number of active neurons per token
        active_ratio = torch.mean(l0_norm) / self.d_sparse
        
        return sparse_latents, {"l0_norm": l0_norm, "active_ratio": active_ratio.item()}

class BDHModel(nn.Module):
    """
    Toy BDH Model (~10M parameters).
    Combines dense embedding with a sparse monosemantic projection layer.
    """
    def __init__(self, vocab_size: int = 1000, d_model: int = 128, d_sparse: int = 2048, k_sparsity: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_sparse = d_sparse
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True
        )
        self.sparse_layer = SparseMonosemanticLayer(d_model=d_model, d_sparse=d_sparse, k_sparsity=k_sparsity)
        self.head = nn.Linear(d_sparse, vocab_size)
        
    def forward(self, input_ids: torch.Tensor):
        """
        Forward pass returning logits and internal sparse layer metrics.
        """
        x = self.embedding(input_ids)
        h = self.transformer_layer(x)
        sparse_latents, metrics = self.sparse_layer(h)
        logits = self.head(sparse_latents)
        return logits, metrics, sparse_latents

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = BDHModel()
    param_count = count_parameters(model)
    print(f"BDH Baseline Toy Model Initialized. Parameter Count: {param_count / 1e6:.2f}M")
    
    # Test forward pass with mock batch
    dummy_input = torch.randint(0, 1000, (2, 16))
    logits, metrics, latents = model(dummy_input)
    print(f"Logits shape: {logits.shape}, Active Ratio: {metrics['active_ratio']:.4f}")
