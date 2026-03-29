"""
Inspect a teacher .pth checkpoint: print all keys/shapes, try loading
into MultiModal, report missing/unexpected keys and suggest rename patterns.

Usage:
    python scripts/inspect_teacher_checkpoint.py outputs/more_pretrained.pth
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.build_model import MultiModal


def inspect(ckpt_path: str):
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and not any(isinstance(v, torch.Tensor) for v in raw.values()):
        print(f"Wrapper keys: {list(raw.keys())}")
        state_dict = raw.get("model_state_dict") or raw.get("state_dict") or raw
    else:
        state_dict = raw

    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    print(f"\n{len(cleaned)} keys in checkpoint. First 30:")
    for k, v in list(cleaned.items())[:30]:
        print(f"  {k:<70s} {str(list(v.shape)):<20s} {v.dtype}")

    print(f"\n{'='*60}")
    model = MultiModal()
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(cleaned.keys())

    matched    = model_keys & ckpt_keys
    missing    = model_keys - ckpt_keys
    unexpected = ckpt_keys  - model_keys

    print(f"Matched:    {len(matched)}")
    print(f"Missing:    {len(missing)}  (model needs, checkpoint lacks)")
    print(f"Unexpected: {len(unexpected)}  (checkpoint has, model ignores)")

    if missing:
        print("\nMissing keys (first 20):")
        for k in sorted(missing)[:20]:
            print(f"  {k}")
    if unexpected:
        print("\nUnexpected keys (first 20):")
        for k in sorted(unexpected)[:20]:
            print(f"  {k}")

    model.load_state_dict(cleaned, strict=False)
    print("\nLoaded with strict=False — ready for use.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/more_pretrained.pth"
    inspect(path)
