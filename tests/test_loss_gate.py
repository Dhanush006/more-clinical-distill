"""
tests/test_loss_gate.py

TDD tests for distill/loss_gate.py — ResidualLossGate.

Run with:
    cd /scratch/user/dshekar/more-clinical-distill
    python -m pytest tests/test_loss_gate.py -v
"""

import sys
sys.path.insert(0, "distill")

import pytest
import torch
import torch.nn as nn

from loss_gate import ResidualLossGate


B = 8          # batch size
EMB = 128      # embedding dim
N_TERMS = 6    # number of loss terms


def make_gate(n_terms=N_TERMS, hidden=64, enabled=True):
    return ResidualLossGate(
        embedding_dim=EMB,
        n_terms=n_terms,
        hidden_dim=hidden,
        enabled=enabled,
    )


def make_inputs(batch=B):
    student = torch.randn(batch, EMB, requires_grad=True)
    t_ecg   = torch.randn(batch, EMB)
    t_cxr   = torch.randn(batch, EMB)
    t_text  = torch.randn(batch, EMB)
    return student, t_ecg, t_cxr, t_text


# ── Shape tests ──────────────────────────────────────────────────────

def test_output_shape_standard_batch():
    gate = make_gate()
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    assert w.shape == (B, N_TERMS), f"Expected ({B},{N_TERMS}), got {w.shape}"


def test_output_shape_batch_1():
    gate = make_gate()
    s, e, c, t = make_inputs(1)
    w = gate(s, e, c, t)
    assert w.shape == (1, N_TERMS)


def test_output_shape_batch_32():
    gate = make_gate()
    s, e, c, t = make_inputs(32)
    w = gate(s, e, c, t)
    assert w.shape == (32, N_TERMS)


def test_custom_n_terms():
    gate = make_gate(n_terms=3)
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    assert w.shape == (B, 3)


# ── Weight validity tests ────────────────────────────────────────────

def test_weights_positive():
    gate = make_gate()
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    assert (w > 0).all(), "All gate weights must be positive (softmax output)"


def test_weights_sum_to_one():
    gate = make_gate()
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    sums = w.sum(dim=1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
        f"Gate weights must sum to 1 per sample, got {sums}"


# ── Stop-gradient test ───────────────────────────────────────────────

def test_stop_gradient_on_student():
    """Gate must not propagate gradients back through the student embedding."""
    gate = make_gate()
    s = torch.randn(B, EMB, requires_grad=True)
    t_ecg  = torch.randn(B, EMB)
    t_cxr  = torch.randn(B, EMB)
    t_text = torch.randn(B, EMB)

    w = gate(s, t_ecg, t_cxr, t_text)
    loss = w.sum()
    loss.backward()

    # s.grad must be None — the gate must detach before using student emb
    assert s.grad is None, (
        "Student embedding gradient must be None after gate backward pass. "
        "Gate input must use s.detach()."
    )


def test_stop_gradient_gate_parameters_receive_grad():
    """Gate parameters must still receive gradients from the weighted loss."""
    gate = make_gate()
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    loss = w.sum()
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in gate.parameters())
    assert has_grad, "Gate parameters must receive gradients"


# ── Entropy regularization tests ─────────────────────────────────────

def test_entropy_positive():
    """Entropy H(w) must be positive for non-degenerate weights."""
    gate = make_gate()
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    H = gate.entropy(w)
    assert H.item() > 0, "Entropy must be positive for non-collapsed weights"


def test_entropy_reg_term_negative():
    """Entropy regularization term (−λ_H × H) must be negative (added to loss to maximize H)."""
    gate = make_gate()
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    reg = gate.entropy_reg(w, lambda_h=0.1)
    assert reg.item() < 0, "Entropy reg term must be negative (encourages high entropy)"


def test_entropy_uniform_is_max():
    """Uniform weights should have entropy >= non-uniform weights."""
    gate = make_gate()
    uniform = torch.ones(B, N_TERMS) / N_TERMS
    collapsed = torch.zeros(B, N_TERMS)
    collapsed[:, 0] = 1.0
    H_uniform   = gate.entropy(uniform)
    H_collapsed = gate.entropy(collapsed + 1e-9)
    assert H_uniform > H_collapsed


# ── Disabled gate (uniform mode) tests ──────────────────────────────

def test_disabled_gate_returns_uniform():
    """When enabled=False, gate must return uniform 1/n weights (ablations A0, A1)."""
    gate = make_gate(enabled=False)
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    expected = torch.ones(B, N_TERMS) / N_TERMS
    assert torch.allclose(w, expected, atol=1e-6), \
        f"Disabled gate must return uniform weights, got {w[0]}"


def test_disabled_gate_no_params():
    """Disabled gate should have no trainable parameters."""
    gate = make_gate(enabled=False)
    n_params = sum(p.numel() for p in gate.parameters())
    assert n_params == 0, f"Disabled gate should have 0 params, got {n_params}"


# ── Device tests ─────────────────────────────────────────────────────

def test_forward_on_cpu():
    gate = make_gate()
    s, e, c, t = make_inputs(B)
    w = gate(s, e, c, t)
    assert w.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_forward_on_cuda():
    gate = make_gate().cuda()
    s = torch.randn(B, EMB, device="cuda")
    e = torch.randn(B, EMB, device="cuda")
    c = torch.randn(B, EMB, device="cuda")
    t = torch.randn(B, EMB, device="cuda")
    w = gate(s, e, c, t)
    assert w.device.type == "cuda"
    assert w.shape == (B, N_TERMS)
