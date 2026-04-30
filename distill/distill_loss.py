"""
distill/distill_loss.py

DistillationLoss — per-sample loss with optional ResidualLossGate weighting.

Loss terms (all reduction='none' → shape (B,)):
    0  L_ecg_BCE    BCE on ECG rhythm logits vs labels
    1  L_pulm_BCE   BCE on pulmonary logits vs labels (-1 uncertain masked out)
    2  L_align_ecg  1 - cosine_sim(student, teacher_ecg)
    3  L_align_cxr  1 - cosine_sim(student, teacher_cxr)       [full-modal only]
    4  L_align_text 1 - cosine_sim(student, teacher_text)      [full-modal only]
    5  L_kl         KL(teacher_probe_logits ‖ student_logits)  [when use_kl=True]

Total loss (corrected gate gradient direction):
    student_component  = mean( w.detach() × terms )   ← student sees fixed routing
    gate_component     = mean( −w × terms.detach() )  ← gate maximises weighted loss
    entropy_reg        = −λ_H × H(w)                  ← prevents gate collapse
    total = student_component + gate_component + entropy_reg

When gate_enabled=False the gate returns uniform 1/n weights.
When modalities=("ecg",) align_cxr and align_text are skipped.

KL term: supply teacher_ecg_logits (computed from a probe on teacher embeddings)
to activate. The probe is a frozen logistic-regression head fit on teacher
embeddings + ECG labels (see scripts/cache_teacher_probe_logits.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_gate import ResidualLossGate

_N_TERMS_FULL     = 5   # ecg, pulm, align_ecg, align_cxr, align_text
_N_TERMS_ECG_ONLY = 3   # ecg, pulm, align_ecg
_N_TERMS_FULL_KL     = 6   # + kl
_N_TERMS_ECG_ONLY_KL = 4   # + kl


class DistillationLoss(nn.Module):
    """
    Args:
        gate_enabled:    use ResidualLossGate (True) or uniform weights (False)
        lambda_h:        entropy regularisation weight (default 0.3)
        kl_temperature:  temperature for soft-label KL distillation (default 4.0)
        modalities:      tuple of modalities to align against; subset of
                         ("ecg", "cxr", "text"). Must include "ecg".
        embedding_dim:   teacher/student embedding dim (default 128)
        gate_hidden_dim: ResidualLossGate hidden layer size (default 64)
        use_kl:          include KL(teacher_probe ‖ student) as an additional
                         loss term. Requires teacher_ecg_logits in forward().
    """

    def __init__(
        self,
        gate_enabled:    bool  = True,
        lambda_h:        float = 0.3,
        kl_temperature:  float = 4.0,
        modalities:      tuple = ("ecg", "cxr", "text"),
        embedding_dim:   int   = 128,
        gate_hidden_dim: int   = 64,
        use_kl:          bool  = False,
    ):
        super().__init__()
        assert "ecg" in modalities, "modalities must include 'ecg'"
        self.lambda_h       = lambda_h
        self.kl_temperature = kl_temperature
        self.modalities     = tuple(modalities)
        self.use_kl         = use_kl

        # Determine number of active terms
        self.use_cxr  = "cxr"  in modalities
        self.use_text = "text" in modalities
        full_modal = self.use_cxr and self.use_text
        if use_kl:
            self.n_terms = _N_TERMS_FULL_KL if full_modal else _N_TERMS_ECG_ONLY_KL
        else:
            self.n_terms = _N_TERMS_FULL if full_modal else _N_TERMS_ECG_ONLY

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
        teacher_ecg_emb:     torch.Tensor,          # (B, 128)
        teacher_cxr_emb:     torch.Tensor,          # (B, 128)
        teacher_text_emb:    torch.Tensor,          # (B, 128)
        ecg_labels:          torch.Tensor,          # (B, 10)
        pulm_labels:         torch.Tensor,          # (B, 4)
        teacher_ecg_logits:  torch.Tensor | None = None,  # (B, 10) probe logits for KL
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

        if self.use_cxr:
            L_align_cxr  = self.align_cxr(student_emb, teacher_cxr_emb)
        else:
            L_align_cxr  = torch.zeros_like(L_ecg)

        if self.use_text:
            L_align_text = self.align_text(student_emb, teacher_text_emb)
        else:
            L_align_text = torch.zeros_like(L_ecg)

        if self.use_kl and teacher_ecg_logits is not None:
            L_kl = self.kl_loss(student_ecg_logits, teacher_ecg_logits)  # (B,)
        else:
            L_kl = torch.zeros_like(L_ecg)

        t_cxr_for_gate  = teacher_cxr_emb  if self.use_cxr  else teacher_ecg_emb
        t_text_for_gate = teacher_text_emb if self.use_text else teacher_ecg_emb

        w = self.gate(student_emb, teacher_ecg_emb, t_cxr_for_gate, t_text_for_gate)  # (B, n_terms)

        full_modal = self.use_cxr and self.use_text
        if self.use_kl:
            if full_modal:
                terms = torch.stack(
                    [L_ecg, L_pulm, L_align_ecg, L_align_cxr, L_align_text, L_kl], dim=1
                )
            else:
                terms = torch.stack([L_ecg, L_pulm, L_align_ecg, L_kl], dim=1)
        else:
            if full_modal:
                terms = torch.stack(
                    [L_ecg, L_pulm, L_align_ecg, L_align_cxr, L_align_text], dim=1
                )
            else:
                terms = torch.stack([L_ecg, L_pulm, L_align_ecg], dim=1)

        # ── Corrected gate gradient direction ────────────────────────
        # Student: minimize over task terms, gate weights act as fixed routing.
        # Gate: maximise the weighted loss (focus on hard terms), minimise negative.
        #   Normalise terms per-sample so the gate sees *relative* difficulty, not
        #   absolute magnitude. Without this, alignment losses (~0.9) always dominate
        #   BCE losses (~0.2) and the gate one-hots onto alignment regardless of λ_H.
        # Entropy reg: prevent gate collapse toward a single term.
        terms_normed      = terms.detach() / terms.detach().sum(dim=1, keepdim=True).clamp(min=1e-9)
        student_component = (w.detach() * terms).sum(dim=1).mean()
        gate_component    = -(w * terms_normed).sum(dim=1).mean()
        entropy           = ResidualLossGate.entropy(w)
        reg               = ResidualLossGate.entropy_reg(w, self.lambda_h)
        total             = student_component + gate_component + reg

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
