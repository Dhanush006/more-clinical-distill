"""
distill/loss_gate.py

ResidualLossGate — dynamically weights 6 distillation loss terms per sample.

Gate input (512-dim):
    z = [sg(s) ‖ sg(s)−t_ecg ‖ sg(s)−t_cxr ‖ sg(s)−t_txt]
    where sg(·) = stop-gradient (.detach())

Gate architecture:
    Linear(512→64) → ReLU → Linear(64→n_terms) → Softmax

Stop-gradient on student is mandatory: the gate reads student state for
routing but must not receive gradient feedback through its input, which
would let the student game the gate rather than learning the task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLossGate(nn.Module):
    """
    Per-sample loss weight allocator for knowledge distillation.

    When enabled=True: learnable MLP that outputs softmax weights over n_terms loss terms.
    When enabled=False: returns uniform 1/n_terms weights (no parameters, for ablations).

    Args:
        embedding_dim: teacher/student embedding dimension (128)
        n_terms:       number of loss terms to weight (default 6)
        hidden_dim:    MLP hidden layer size (default 64)
        enabled:       if False, returns uniform weights (ablation mode)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        n_terms:       int = 6,
        hidden_dim:    int = 64,
        enabled:       bool = True,
    ):
        super().__init__()
        self.n_terms = n_terms
        self.enabled = enabled

        if enabled:
            input_dim = embedding_dim * 4  # [s ‖ s−ecg ‖ s−cxr ‖ s−txt]
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_terms),
            )
        # If disabled, no parameters registered — optimizer sees nothing to update.

    def forward(
        self,
        student_emb:   torch.Tensor,   # (B, 128)
        teacher_ecg:   torch.Tensor,   # (B, 128)
        teacher_cxr:   torch.Tensor,   # (B, 128)
        teacher_text:  torch.Tensor,   # (B, 128)
    ) -> torch.Tensor:                 # (B, n_terms)
        if not self.enabled:
            B = student_emb.size(0)
            return torch.ones(B, self.n_terms, device=student_emb.device) / self.n_terms

        # Stop-gradient: gate observes student state but cannot train it to fool routing.
        s = student_emb.detach()

        z = torch.cat([
            s,
            s - teacher_ecg,
            s - teacher_cxr,
            s - teacher_text,
        ], dim=-1)                    # (B, 512)

        logits = self.mlp(z)          # (B, n_terms)
        return F.softmax(logits, dim=-1)

    # ── Entropy helpers ──────────────────────────────────────────────

    @staticmethod
    def entropy(weights: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Mean batch entropy H(w) = -Σ w_i log(w_i). Returns scalar."""
        return -(weights * torch.log(weights + eps)).sum(dim=-1).mean()

    @staticmethod
    def entropy_reg(weights: torch.Tensor, lambda_h: float = 0.1, eps: float = 1e-9) -> torch.Tensor:
        """−λ_H × H(w). Subtract from total loss to maximise entropy (prevent collapse)."""
        H = -(weights * torch.log(weights + eps)).sum(dim=-1).mean()
        return -lambda_h * H
