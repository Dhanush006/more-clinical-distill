#!/usr/bin/env python3
"""
scripts/run_smoke_test.py

Validates the full MoRE forward pass on the smoke-test .npy subset.

Checks (in order):
  1. .npy loads, items have correct shape
  2. ECG loads via wfdb, preprocesses to (12, 1000) float tensor
  3. CXR loads via PIL, preprocesses to (3, 224, 224) float tensor
  4. Notes tokenize to (512,) via RoBERTa tokenizer
  5. One forward pass through MultiModal on a single batch succeeds
  6. InfoNCE loss is finite (not NaN)

Usage:
    python scripts/run_smoke_test.py [--config configs/paths.yaml]
                                     [--smoke-config configs/smoke_test.yaml]
                                     [--data data/processed/smoke_test.npy]
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [PASS] {label}")
        return result
    except Exception as e:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MoRE smoke test")
    parser.add_argument("--config",       default="configs/paths.yaml")
    parser.add_argument("--smoke-config", default="configs/smoke_test.yaml")
    parser.add_argument("--data",         default=None,
                        help="Override smoke test .npy path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.smoke_config) as f:
        scfg = yaml.safe_load(f)

    data_path = args.data or scfg["data_path"]

    import sys
    sys.path.append("./utils")

    print("=" * 60)
    print("MoRE Smoke Test")
    print("=" * 60)

    # ── 1. Load .npy ──────────────────────────────────────────────────
    data = check("Load smoke .npy", lambda: np.load(data_path, allow_pickle=True))
    check("Items have 6 elements", lambda: (
        all(len(item) == 6 for item in data[:5]) or
        (_ for _ in ()).throw(ValueError(f"Expected 6, got {len(data[0])}"))
    ))
    print(f"     {len(data)} items loaded")

    # ── 2. ECG load ───────────────────────────────────────────────────
    def ecg_check():
        import wfdb
        import torch
        from scipy.signal import resample_poly
        stem = data[0][1]
        sig, fields = wfdb.rdsamp(stem)
        assert sig.shape[1] == 12, f"Expected 12 leads, got {sig.shape[1]}"
        sig = np.nan_to_num(sig, nan=0.0)
        sig = resample_poly(sig, up=1, down=5).T      # (12, ~1000)
        sig = sig[:, :1000]
        assert sig.shape == (12, 1000), f"Shape mismatch: {sig.shape}"
        return torch.FloatTensor(sig)

    ecg_tensor = check("ECG: wfdb load + resample → (12,1000)", ecg_check)
    print(f"     shape={tuple(ecg_tensor.shape)}, dtype={ecg_tensor.dtype}")

    # ── 3. CXR load ───────────────────────────────────────────────────
    def cxr_check():
        import torch
        import torchvision.transforms as transforms
        from PIL import Image
        import cv2

        xray_path = data[0][0]
        img = Image.open(xray_path).convert("L")
        img = img.resize((224, 224))
        img_np = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq = clahe.apply(img_np)
        t = transforms.ToTensor()(Image.fromarray(img_eq))  # (1,224,224)
        t = torch.cat([t, t, t], dim=0)                     # (3,224,224)
        assert t.shape == (3, 224, 224), f"Shape: {t.shape}"
        return t

    cxr_tensor = check("CXR: PIL load + CLAHE → (3,224,224)", cxr_check)
    print(f"     shape={tuple(cxr_tensor.shape)}")

    # ── 4. Tokenizer ─────────────────────────────────────────────────
    def tok_check():
        import os
        from transformers import RobertaTokenizerFast
        roberta_path = os.environ.get("MORE_ROBERTA_PATH", cfg.get("roberta_path",
                       "./data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf"))
        tokenizer = RobertaTokenizerFast.from_pretrained(roberta_path)
        note = data[0][2] + " " + data[0][3]
        enc = tokenizer(note, max_length=512, padding="max_length",
                        truncation=True, return_tensors="pt")
        assert enc["input_ids"].shape == (1, 512)
        return tokenizer

    tokenizer = check("Notes: tokenize to (512,)", tok_check)

    # ── 5 + 6. Forward pass ───────────────────────────────────────────
    def forward_check():
        import os, torch
        from torch.cuda.amp import autocast
        from info_nce import InfoNCE
        from create_dataset import MultiModalData
        from build_model import MultiModal
        from torch.utils.data import DataLoader
        from ecg_augmentations import ECGAugmentor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"     device={device}")

        augmentor = ECGAugmentor()
        dataset = MultiModalData(list(data[:scfg["batch_size"]]), augmentor)
        loader = DataLoader(dataset, batch_size=scfg["batch_size"],
                            num_workers=0, shuffle=False)

        roberta_path = os.environ.get(
            "MORE_ROBERTA_PATH",
            cfg.get("roberta_path", "./data/RoBERTa-base-PM-M3/RoBERTa-base-PM-M3-hf")
        )
        os.environ["MORE_ROBERTA_PATH"] = roberta_path

        model = MultiModal().to(device)
        model.eval()

        criterion = InfoNCE(scfg["temperature_initial"])

        with torch.no_grad():
            xray, ecg, note_id, mask = next(iter(loader))
            xray  = xray.to(device)
            ecg   = ecg.to(device)
            note_id = note_id.to(device)
            mask  = mask.to(device)
            with autocast():
                xray_out, xray_text, ecg_out, ecg_text, note_out = model(
                    xray, ecg, note_id, mask
                )
                loss = (
                    (criterion(note_out, xray_text) + criterion(xray_text, note_out)) / 2 +
                    (criterion(note_out, ecg_text)  + criterion(ecg_text,  note_out)) / 2
                )
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        return loss.item()

    loss_val = check("Forward pass + InfoNCE loss is finite", forward_check)
    print(f"     loss={loss_val:.4f}")

    print()
    print("=" * 60)
    print("ALL CHECKS PASSED — MoRE pipeline is smoke-test ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
