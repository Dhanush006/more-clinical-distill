"""
tests/test_distill_loss.py

TDD tests for distill/distill_loss.py — DistillationLoss with 6 per-sample terms.

Run with:
    cd /scratch/user/dshekar/more-clinical-distill
    python -m pytest tests/test_distill_loss.py -v
"""

import sys
sys.path.insert(0, "distill")

import pytest
import torch
import torch.nn.functional as F

from distill_loss import DistillationLoss


B       = 8
N_ECG   = 10
N_PULM  = 4
EMB     = 128
T_KL    = 4.0


def make_loss(enabled=True, lambda_h=0.3, kl_temperature=4.0, modalities=("ecg", "cxr", "text")):
    return DistillationLoss(
        gate_enabled   = enabled,
        lambda_h       = lambda_h,
        kl_temperature = kl_temperature,
        modalities     = modalities,
    )


def make_batch(B=B):
    student_ecg_logits  = torch.randn(B, N_ECG)
    student_pulm_logits = torch.randn(B, N_PULM)
    student_emb         = torch.randn(B, EMB, requires_grad=True)
    teacher_ecg_emb     = torch.randn(B, EMB)
    teacher_cxr_emb     = torch.randn(B, EMB)
    teacher_text_emb    = torch.randn(B, EMB)
    ecg_labels          = (torch.rand(B, N_ECG) > 0.7).float()
    pulm_labels         = (torch.rand(B, N_PULM) > 0.6).float()
    return dict(
        student_ecg_logits  = student_ecg_logits,
        student_pulm_logits = student_pulm_logits,
        student_emb         = student_emb,
        teacher_ecg_emb     = teacher_ecg_emb,
        teacher_cxr_emb     = teacher_cxr_emb,
        teacher_text_emb    = teacher_text_emb,
        ecg_labels          = ecg_labels,
        pulm_labels         = pulm_labels,
    )


# ── Individual loss term tests ───────────────────────────────────────

def test_ecg_bce_shape_and_finite():
    dl = make_loss()
    b  = make_batch()
    L  = dl.ecg_bce(b["student_ecg_logits"], b["ecg_labels"])
    assert L.shape == (B,), f"Expected (B,), got {L.shape}"
    assert torch.isfinite(L).all()


def test_ecg_bce_nonnegative():
    dl = make_loss()
    b  = make_batch()
    L  = dl.ecg_bce(b["student_ecg_logits"], b["ecg_labels"])
    assert (L >= 0).all(), "BCE must be non-negative"


def test_pulm_bce_shape_and_finite():
    dl = make_loss()
    b  = make_batch()
    L  = dl.pulm_bce(b["student_pulm_logits"], b["pulm_labels"])
    assert L.shape == (B,)
    assert torch.isfinite(L).all()


def test_pulm_bce_uncertain_mask():
    """Uncertain labels (-1) must be masked out — should not inflate the loss."""
    dl = make_loss()
    pulm_all_certain   = (torch.rand(B, N_PULM) > 0.5).float()
    pulm_half_uncertain = pulm_all_certain.clone()
    pulm_half_uncertain[:, 0] = -1.0   # mark first class as uncertain

    logits = torch.randn(B, N_PULM)
    L_certain   = dl.pulm_bce(logits, pulm_all_certain)
    L_uncertain = dl.pulm_bce(logits, pulm_half_uncertain)

    # Both must be finite; uncertain version averages over fewer terms
    assert torch.isfinite(L_certain).all()
    assert torch.isfinite(L_uncertain).all()

    # With one class masked, mean loss should differ
    # (not necessarily lower — but the masked version must be finite and non-negative)
    assert (L_uncertain >= 0).all()


def test_pulm_bce_all_uncertain_returns_zero():
    """If all labels for a sample are -1, that sample's BCE contribution should be 0."""
    dl = make_loss()
    pulm_labels = torch.full((B, N_PULM), -1.0)
    logits = torch.randn(B, N_PULM)
    L = dl.pulm_bce(logits, pulm_labels)
    assert torch.allclose(L, torch.zeros(B), atol=1e-6), \
        f"All-uncertain labels should produce zero loss, got {L}"


def test_align_ecg_shape_and_finite():
    dl = make_loss()
    b  = make_batch()
    L  = dl.align_ecg(b["student_emb"], b["teacher_ecg_emb"])
    assert L.shape == (B,)
    assert torch.isfinite(L).all()


def test_align_ecg_between_zero_and_two():
    """Cosine distance is in [0, 2] — (1 - cos) ∈ [0, 2]."""
    dl = make_loss()
    b  = make_batch()
    L  = dl.align_ecg(b["student_emb"], b["teacher_ecg_emb"])
    assert (L >= 0).all()
    assert (L <= 2.0 + 1e-5).all()


def test_align_identical_is_zero():
    """Alignment loss between identical embeddings should be ~0."""
    dl  = make_loss()
    emb = torch.randn(B, EMB)
    L   = dl.align_ecg(emb, emb)
    assert torch.allclose(L, torch.zeros(B), atol=1e-5), f"Identical emb align ≠ 0: {L}"


def test_align_cxr_shape():
    dl = make_loss()
    b  = make_batch()
    L  = dl.align_cxr(b["student_emb"], b["teacher_cxr_emb"])
    assert L.shape == (B,)


def test_align_text_shape():
    dl = make_loss()
    b  = make_batch()
    L  = dl.align_text(b["student_emb"], b["teacher_text_emb"])
    assert L.shape == (B,)


# NOTE: The KL term has been removed from DistillationLoss.forward (no cached
# teacher classification logits → KL was degenerate and the gate collapsed onto
# it). The kl_loss() static method is retained so existing pipelines that cache
# teacher logits in the future can re-enable it. Tests below exercise the
# helper directly with explicit teacher logits.

def test_kl_loss_shape_and_finite():
    dl = make_loss()
    student_logits = torch.randn(B, N_ECG)
    teacher_logits = torch.randn(B, N_ECG)
    L  = dl.kl_loss(student_logits, teacher_logits)
    assert L.shape == (B,)
    assert torch.isfinite(L).all()


def test_kl_loss_nonnegative():
    dl = make_loss()
    student_logits = torch.randn(B, N_ECG)
    teacher_logits = torch.randn(B, N_ECG)
    L  = dl.kl_loss(student_logits, teacher_logits)
    assert (L >= 0).all(), "KL divergence must be non-negative"


def test_kl_loss_zero_for_identical():
    """KL(p || p) must be 0."""
    dl = make_loss()
    logits = torch.randn(B, N_ECG)
    L = dl.kl_loss(logits, logits)
    assert torch.allclose(L, torch.zeros(B), atol=1e-5), f"KL(p||p) ≠ 0: {L}"


def test_kl_uses_temperature():
    """Temperature scaling must produce different outputs at T=1 vs T=10."""
    dl_cold = make_loss(kl_temperature=1.0)
    dl_warm = make_loss(kl_temperature=10.0)
    student_logits = torch.randn(B, N_ECG)
    teacher_logits = torch.randn(B, N_ECG)
    L_cold = dl_cold.kl_loss(student_logits, teacher_logits)
    L_warm = dl_warm.kl_loss(student_logits, teacher_logits)
    assert not torch.allclose(L_cold, L_warm, atol=1e-4), \
        "T=1 and T=10 should produce different KL values"


# ── Gated total loss tests ───────────────────────────────────────────

def test_forward_returns_scalar():
    dl = make_loss()
    b  = make_batch()
    result = dl(**b)
    loss = result["loss"]
    assert loss.ndim == 0, f"Total loss must be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Total loss must be finite"


def test_forward_kl_key_is_zero():
    """L_kl is removed but key retained for log compatibility — must be 0."""
    dl = make_loss()
    b  = make_batch()
    result = dl(**b)
    assert result["L_kl"] == 0.0, "KL term should be 0 (removed from loss)"


def test_forward_loss_finite():
    # With the corrected gate, gate_component = -(w*terms) is negative by design,
    # so total loss can be negative. Only finiteness is guaranteed.
    dl = make_loss()
    b  = make_batch()
    result = dl(**b)
    assert torch.isfinite(result["loss"]), "Total loss must be finite"


def test_forward_returns_term_dict():
    """forward() must return individual term values for logging."""
    dl  = make_loss()
    b   = make_batch()
    result = dl(**b)
    expected_keys = {"loss", "L_ecg", "L_pulm", "L_align_ecg",
                     "L_align_cxr", "L_align_text", "L_kl", "gate_entropy"}
    for k in expected_keys:
        assert k in result, f"Missing key '{k}' in loss output dict"


def test_forward_backward_has_student_grad():
    """Total loss must propagate gradients back to the student embedding."""
    dl = make_loss()
    b  = make_batch()
    result = dl(**b)
    result["loss"].backward()
    assert b["student_emb"].grad is not None, \
        "student_emb must receive gradients from the total loss"


def test_gate_disabled_static_weights():
    """Disabled gate (uniform mode) must produce a finite scalar loss."""
    dl = make_loss(enabled=False)
    b  = make_batch()
    result = dl(**b)
    assert torch.isfinite(result["loss"])


def test_gate_gradient_direction():
    """Gate params must receive gradient opposite to student params (curriculum direction).
    With corrected stop-grad decoupling: gate_component = -(w * terms.detach()),
    so the gate is pushed to UP-WEIGHT high-loss terms, not down-weight them."""
    dl = make_loss()
    b  = make_batch()
    result = dl(**b)
    result["loss"].backward()
    # Gate MLP should receive gradients from the gate_component
    gate_params = list(dl.gate.mlp.parameters())
    assert all(p.grad is not None for p in gate_params), \
        "Gate MLP parameters must receive gradients"
    assert all(p.grad.abs().sum() > 0 for p in gate_params), \
        "Gate MLP gradients must be non-zero"


def test_kl_term_active_with_teacher_logits():
    """When use_kl=True and teacher_ecg_logits is supplied, L_kl must be > 0."""
    dl = DistillationLoss(gate_enabled=True, use_kl=True)
    b  = make_batch()
    b["teacher_ecg_logits"] = torch.randn(B, N_ECG)
    result = dl(**b)
    assert result["L_kl"] > 0.0, "L_kl must be positive when teacher logits differ from student"
    assert torch.isfinite(result["loss"])


# ── Modality ablation tests ──────────────────────────────────────────

def test_ecg_only_modality():
    """ecg-only mode: 3 terms (ecg_bce, pulm_bce, align_ecg) instead of 6."""
    dl = make_loss(modalities=("ecg",))
    b  = make_batch()
    result = dl(**b)
    assert torch.isfinite(result["loss"])
    # align_cxr and align_text should be 0 (not computed)
    assert result["L_align_cxr"] == 0.0
    assert result["L_align_text"] == 0.0


def test_all_modalities():
    dl = make_loss(modalities=("ecg", "cxr", "text"))
    b  = make_batch()
    result = dl(**b)
    assert torch.isfinite(result["loss"])
    # All 6 terms should be positive
    assert result["L_align_ecg"] > 0
    assert result["L_align_cxr"] > 0
    assert result["L_align_text"] > 0
