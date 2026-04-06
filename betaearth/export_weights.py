"""Export model weights from training checkpoints to standalone .pt files.

Extracts individual models from multi-model training checkpoints and saves
them as clean state dicts loadable by BetaEarth.from_pretrained().

Usage:
    python betaearth_hf/export_weights.py
"""

import torch
from pathlib import Path

EXPORT_DIR = Path("betaearth_hf/weights")


def export_frozen_variants():
    """Export reinit_fusion and hilr_fusion from frozen variants checkpoint."""
    ckpt_path = "checkpoints/segformer_film_frozen_best.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    for model_attr, export_name in [
        ("model_reinit", "betaearth-segformer-film"),
        ("model_hilr", "betaearth-segformer-film-hilr"),
    ]:
        prefix = f"{model_attr}."
        model_state = {k[len(prefix):]: v for k, v in state.items()
                       if k.startswith(prefix)}
        out_path = EXPORT_DIR / export_name / "model.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, out_path)
        n_params = sum(v.numel() for v in model_state.values())
        print(f"Exported {export_name}: {len(model_state)} tensors, {n_params/1e6:.1f}M params -> {out_path}")


def export_isprs_segformer():
    """Export SegFormer-B2 (no FiLM) from ISPRS checkpoint."""
    ckpt_path = "checkpoints/multi_final.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    prefix = "models.segformer_b2."
    model_state = {k[len(prefix):]: v for k, v in state.items()
                   if k.startswith(prefix)}
    out_path = EXPORT_DIR / "betaearth-segformer" / "model.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, out_path)
    n_params = sum(v.numel() for v in model_state.values())
    print(f"Exported betaearth-segformer: {len(model_state)} tensors, {n_params/1e6:.1f}M params -> {out_path}")


def export_triple_variants():
    """Export scratch and rgb_only from triple SegFormer checkpoint."""
    ckpt_path = "checkpoints/segformer_film_scratch_best.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    for model_attr, export_name in [
        ("model_scratch", "betaearth-segformer-film-scratch"),
        ("model_rgb", "betaearth-rgb-only"),
    ]:
        prefix = f"{model_attr}."
        model_state = {k[len(prefix):]: v for k, v in state.items()
                       if k.startswith(prefix)}
        if not model_state:
            print(f"Warning: no keys found for {model_attr}")
            continue
        out_path = EXPORT_DIR / export_name / "model.pt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_state, out_path)
        n_params = sum(v.numel() for v in model_state.values())
        print(f"Exported {export_name}: {len(model_state)} tensors, {n_params/1e6:.1f}M params -> {out_path}")


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print("Exporting model weights...\n")

    export_frozen_variants()
    export_isprs_segformer()
    export_triple_variants()

    print(f"\nAll weights exported to {EXPORT_DIR}/")
    print("\nTo upload to HuggingFace Hub:")
    print("  huggingface-cli upload asterisk-labs/betaearth-segformer-film betaearth_hf/weights/betaearth-segformer-film/")


if __name__ == "__main__":
    main()
