"""
distill/distill_loss.py

DistillationLoss — 6-term per-sample loss with optional ResidualLossGate weighting.

Loss terms (all reduction='none' → shape (B,)):
    0  L_ecg_BCE    BCE on ECG rhythm logits vs labels
    1  L_pulm_BCE   BCE on pulmonary logits vs labels (-1 uncertain masked out)
    2  L_align_ecg  1 - cosine_sim(student, teacher_ecg)
    3  L_align_cxr  1 - cosine_sim(student, teacher_cxr)
    4  L_align_text 1 - cosine_sim(student, teacher_text)
    5  L_kl         KL(softmax(student/T) ‖ softmax(teacher/T))

Total loss:
    For each sample i:
        L_total[i] = Σ_k  w[i,k] × term_k[i]
    Scalar: mean(L_total) + entropy_reg(w)

When gate_enabled=False the gate returns uniform 1/6 weights.
When modalities=("ecg",) align_cxr and align_text are skipped (gate uses 4 terms).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_gate import ResidualLossGate

# Indices of each term in the weight vector
_IDX = {"ecg": 0, "pulm": 1, "align_ecg": 2, "align_cxr": 3, "align_text": 4, "kl": 5}
_N_TERMS_FULL = 6
_N_TERMS_ECG_ONLY = 3   # ecg, pulm, align_ecg (no cxr/text)


class DistillationLoss(nn.Module):
    """
    Args:
        gate_enabled:    use ResidualLossGate (True) or uniform weights (False)
        lambda_h:        entropy regularisation weight (default 0.1)
        kl_temperature:  temperature for soft-label KL distillation (default 4.0)
        modalities:      tuple of modalities to align against; subset of
                         ("ecg", "cxr", "text"). Must include "ecg".
        embedding_dim:   teacher/student embedding dim (default 128)
        gate_hidden_dim: ResidualLossGate hidden layer size (default 64)
    """

    def __init__(
        self,
        gate_enabled:    bool  = True,
        lambda_h:        float = 0.1,
        kl_temperature:  float = 4.0,
        modalities:      tuple = ("ecg", "cxr", "text"),
        embedding_dim:   int   = 128,
        gate_hidden_dim: int   = 64,
    ):
        super().__init__()
        assert "ecg" in modalities, "modalities must include 'ecg'"
        self.lambda_h       = lambda_h
        self.kl_temperature = kl_temperature
        self.modalities     = tuple(modalities)

        # Determine number of active terms
        self.use_cxr  = "cxr"  in modalities
        self.use_text = "text" in modalities
        self.n_terms  = _N_TERMS_FULL if (self.use_cxr and self.use_text) else _N_TERMS_ECG_ONLY

        self.gate = ResidualLossGate(
            embedding_dim = embedding_dim,
            n_terms       = self.n_terms,
            hidden_dim    = gate_hidden_dim,
            enabled       = gate_enabled,
        )

    # ── Individual per-sample loss terms ────────────────────────────

    @staticmethod
    def ecg_bce(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """BCE per sample, shape (B,). Mean over classes per sample."""
        return F.binary_cross_entropy_with_logits(
            student_logits, labels.clamp(min=0.0), reduction="none"
        ).mean(dim=-1)

    @staticmethod
    def pulm_bce(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """BCE per sample with uncertain-label (-1) masking. Shape (B,)."""
        mask = (labels != -1.0).float()       # (B, 4)
        bce  = F.binary_cross_entropy_with_logits(
            student_logits, labels.clamp(min=0.0), reduction="none"
        )                                      # (B, 4)
        masked = bce * mask
        denom  = mask.sum(dim=-1).clamp(min=1.0)
        return masked.sum(dim=-1) / denom      # (B,)

    @staticmethod
    def align_ecg(student_emb: torch.Tensor, teacher_ecg: torch.Tensor) -> torch.Tensor:
        """1 − cosine_sim per sample. Shape (B,)."""
        return 1.0 - F.cosine_similarity(student_emb, teacher_ecg, dim=-1)

    @staticmethod
    def align_cxr(student_emb: torch.Tensor, teacher_cxr: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(student_emb, teacher_cxr, dim=-1)

    @staticmethod
    def align_text(student_emb: torch.Tensor, teacher_text: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(student_emb, teacher_text, dim=-1)

    def kl_loss(
        self,
        student_logits:  torch.Tensor,
        teacher_logits:  torch.Tensor,
    ) -> torch.Tensor:
        """KL(p_teacher ‖ p_student) per sample after temperature scaling. Shape (B,)."""
        T = self.kl_temperature
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
        p_student = F.log_softmax(student_logits / T, dim=-1)
        # kl_div: input=log_prob, target=prob, returns per-element; sum over classes
        return F.kl_div(p_student, p_teacher, reduction="none").sum(dim=-1) * (T * T)

    # ── Gated total loss ─────────────────────────────────────────────

    def forward(
        self,
        student_ecg_logits:  torch.Tensor,          # (B, 10)
        student_pulm_logits: torch.Tensor,          # (B, 4)
        student_emb:         torch.Tensor,          # (B, 128)
        teacher_ecg_logits:  torch.Tensor,          # (B, 10)
        teacher_ecg_emb:     torch.Tensor,          # (B, 128)
        teacher_cxr_emb:     torch.Tensor,          # (B, 128)
        teacher_text_emb:    torch.Tensor,          # (B, 128)
        ecg_labels:          torch.Tensor,          # (B, 10)
        pulm_labels:         torch.Tensor,          # (B, 4)
    ) -> dict:
        """
        Returns:
            dict with keys: loss (scalar), L_ecg, L_pulm, L_align_ecg,
                            L_align_cxr, L_align_text, L_kl, gate_entropy
        """
        # ── Per-sample terms ─────────────────────────────────────────
        L_ecg       = self.ecg_bce(student_ecg_logits, ecg_labels)      # (B,)
        L_pulm      = self.pulm_bce(student_pulm_logits, pulm_labels)   # (B,)
        L_align_ecg = self.align_ecg(student_emb, teacher_ecg_emb)      # (B,)
        L_kl        = self.kl_loss(student_ecg_logits, teacher_ecg_logits)  # (B,)

        if self.use_cxr:
            L_align_cxr  = self.align_cxr(student_emb, teacher_cxr_emb)
        else:
            L_align_cxr  = torch.zeros_like(L_ecg)

        if self.use_text:
            L_align_text = self.align_text(student_emb, teacher_text_emb)
        else:
            L_align_text = torch.zeros_like(L_ecg)

        # ── Gate weights ─────────────────────────────────────────────
        # Use only the active modality embeddings for gate input
        t_cxr_for_gate  = teacher_cxr_emb  if self.use_cxr  else teacher_ecg_emb
        t_text_for_gate = teacher_text_emb if self.use_text else teacher_ecg_emb

        w = self.gate(student_emb, teacher_ecg_emb, t_cxr_for_gate, t_text_for_gate)  # (B, n_terms)

        # ── Stack active terms ───────────────────────────────────────
        if self.n_terms == _N_TERMS_FULL:
            terms = torch.stack([L_ecg, L_pulm, L_align_ecg, L_align_cxr, L_align_text, L_kl], dim=1)
        else:
            terms = torch.stack([L_ecg, L_pulm, L_align_ecg], dim=1)  # (B, 3)

        # ── Gated scalar loss ────────────────────────────────────────
        gated     = (w * terms).sum(dim=1)              # (B,)
        base_loss = gated.mean()
        entropy   = ResidualLossGate.entropy(w)
        reg       = ResidualLossGate.entropy_reg(w, self.lambda_h)  # negative (maximise H)
        total     = base_loss + reg

        return {
            "loss":          total,
            "L_ecg":         L_ecg.mean().item(),
            "L_pulm":        L_pulm.mean().item(),
            "L_align_ecg":   L_align_ecg.mean().item(),
            "L_align_cxr":   L_align_cxr.mean().item(),
            "L_align_text":  L_align_text.mean().item(),
            "L_kl":          L_kl.mean().item(),
            "gate_entropy":  entropy.item(),
        }
