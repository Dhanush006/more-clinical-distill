"""
distill/student_model.py

MobileNetV3-Small adapted for single-lead 1-D ECG classification.

Architecture:
  - Input: (B, 1, T) single-lead ECG  (T=1000 at 100 Hz, or raw 5000 at 500 Hz)
  - Stem: Conv1d projection → 3-channel pseudo-image reshape trick, or
           direct 1-D feature extractor using depthwise-separable convs
           mirroring MobileNetV3-Small structure
  - Head: Linear(576 → num_classes=4)
  - Output: logits (B, 4) + embedding (B, embedding_dim)

Parameters: ~2.54M  (MobileNetV3-Small backbone)

NOTE: This is a scaffold. The forward pass shape logic is implemented;
      the distillation loss connection is wired in distill_train.py.
"""

import torch
import torch.nn as nn
import timm


class SingleLeadECGStudent(nn.Module):
    """
    MobileNetV3-Small for single-lead ECG.

    Converts (B, 1, T) ECG to a 2-D representation by treating the
    1-D signal as a 1×T image, then uses a standard MobileNetV3-Small
    feature extractor.  A learned 1-D projection adapts the single
    lead into 3 channels before entering the backbone.
    """

    def __init__(
        self,
        num_classes: int = 4,
        embedding_dim: int = 128,
        signal_length: int = 1000,
        pretrained: bool = False,
    ):
        super().__init__()
        self.signal_length = signal_length

        # ── 1-D → 3-channel adapter ───────────────────────────────────
        # Input:  (B, 1, T)
        # Output: (B, 3, T)
        self.channel_adapter = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.Hardswish(),
            nn.Conv1d(16, 3, kernel_size=1, bias=False),
        )

        # ── MobileNetV3-Small backbone (2-D, adapted for 1×T input) ──
        # We reshape (B, 3, T) → (B, 3, 1, T) so standard 2-D convs work.
        # timm's mobilenetv3_small_100 expects (B, 3, H, W).
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0,          # remove classifier head
            global_pool="avg",
        )
        backbone_dim = self.backbone.num_features  # 576 for small_100

        # ── Projection head (matches teacher ECG projector dim=128) ───
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

        # ── Classification head ───────────────────────────────────────
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, T)  single-lead ECG at 100 Hz

        Returns:
            logits:    (B, num_classes)
            embedding: (B, embedding_dim)  — used for distillation loss
        """
        # (B, 1, T) → (B, 3, T)
        x = self.channel_adapter(x)

        # (B, 3, T) → (B, 3, 1, T) for 2-D backbone
        x = x.unsqueeze(2)

        # (B, 3, 1, T) → (B, backbone_dim) via MobileNetV3-Small
        features = self.backbone(x)   # global avg pool built in

        # embedding + logits
        embedding = self.projector(features)
        logits    = self.classifier(embedding)

        return logits, embedding


def build_student(cfg: dict) -> SingleLeadECGStudent:
    """Construct student from distill_config.yaml student block."""
    return SingleLeadECGStudent(
        num_classes=cfg.get("num_classes", 4),
        embedding_dim=cfg.get("embedding_dim", 128),
        signal_length=cfg.get("signal_length", 1000),
        pretrained=cfg.get("pretrained_backbone", False),
    )
