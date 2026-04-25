"""
distill/student_model.py

MobileNetV3-Small adapted for dual-head single-lead ECG classification.

Architecture:
  - Input: (B, 1, T) single-lead ECG  (T=1000 at 100 Hz)
  - Channel adapter: Conv1d 1→3 channels
  - Reshape: (B,3,T) → (B,3,25,40) near-square 2-D representation
  - Backbone: MobileNetV3-Small (timm) with global avg pool → backbone_dim features
  - Projector: Linear → 256 → ReLU → 128-dim shared embedding
  - Head 1 (primary):   ecg_classifier  Linear(128 → ecg_num_classes=10)
                        ECG rhythm classes: [Normal, Sinus bradycardia, Sinus tachycardia,
                        Atrial fibrillation, LBBB, RBBB, ST elevation MI,
                        ST ischemia, AV block, LVH]
  - Head 2 (secondary): pulm_classifier Linear(128 → pulm_num_classes=4)
                        Pulmonary labels:  [Atelectasis, Cardiomegaly, Edema, Pleural Effusion]
  - Output: ecg_logits (B,10), pulm_logits (B,4), embedding (B,128)

Parameters: ~1.82M
"""

import torch
import torch.nn as nn
import timm


class SingleLeadECGStudent(nn.Module):
    """
    MobileNetV3-Small for single-lead ECG with dual classification heads.

    Primary head:   ECG rhythm classification (10 classes, multi-label)
    Secondary head: Pulmonary pathology from ECG via distillation (4 classes)

    Both heads share the same backbone and 128-dim embedding.
    """

    def __init__(
        self,
        ecg_num_classes: int = 10,
        pulm_num_classes: int = 4,
        embedding_dim: int = 128,
        signal_length: int = 1000,
        pretrained: bool = False,
        backbone_name: str = "mobilenetv3_small_100",
    ):
        super().__init__()
        self.signal_length = signal_length
        self.backbone_name = backbone_name

        # ── 1-D → 3-channel adapter ───────────────────────────────────
        self.channel_adapter = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.Hardswish(),
            nn.Conv1d(16, 3, kernel_size=1, bias=False),
        )

        # ── 2-D backbone (timm: mobilenetv3_small_100 / efficientnet_b0 / resnet18 / etc.)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        # Detect actual output dim via a dummy forward (timm reports 576 via
        # num_features but the conv_head brings it to 1024 before global pool)
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, 25, 40)
            backbone_dim = self.backbone(_dummy).shape[1]

        # ── Projection head (matches teacher ECG projector dim=128) ───
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

        # ── Primary head: ECG rhythm classification (10 classes) ─────
        self.ecg_classifier = nn.Linear(embedding_dim, ecg_num_classes)

        # ── Secondary head: pulmonary pathology via distillation ──────
        self.pulm_classifier = nn.Linear(embedding_dim, pulm_num_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 1, T)  single-lead ECG at 100 Hz

        Returns:
            ecg_logits:  (B, ecg_num_classes)   primary ECG rhythm head
            pulm_logits: (B, pulm_num_classes)  secondary pulmonary head
            embedding:   (B, embedding_dim)     shared embedding (for distillation)
        """
        # (B, 1, T) → (B, 3, T)
        x = self.channel_adapter(x)

        # (B, 3, T) → (B, 3, 25, 40) — near-square 2-D for MobileNetV3-Small
        B = x.shape[0]
        x = x.reshape(B, 3, 25, 40)

        # backbone + global avg pool → (B, backbone_dim)
        features = self.backbone(x)

        # shared embedding
        embedding = self.projector(features)

        # dual heads
        ecg_logits  = self.ecg_classifier(embedding)
        pulm_logits = self.pulm_classifier(embedding)

        return ecg_logits, pulm_logits, embedding


def build_student(cfg: dict) -> SingleLeadECGStudent:
    """Construct student from distill_config.yaml student block."""
    return SingleLeadECGStudent(
        ecg_num_classes=cfg.get("ecg_num_classes", 10),
        pulm_num_classes=cfg.get("pulm_num_classes", 4),
        embedding_dim=cfg.get("embedding_dim", 128),
        signal_length=cfg.get("signal_length", 1000),
        pretrained=cfg.get("pretrained_backbone", False),
        backbone_name=cfg.get("backbone", "mobilenetv3_small_100"),
    )
