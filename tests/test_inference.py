# tests/test_inference.py
"""
Standalone inference test script for the fine-tuned Stable Diffusion model.

What it does:
- Loads the local Diffusers pipeline from: models/best_evaluated_model
- Generates 1 image from a prompt (fast default settings)
- Saves output to: outputs/test_output.png
- Prints device + timing

Run:
  python tests/test_inference.py
Optional:
  python tests/test_inference.py --device cuda --steps 20 --cfg 7.5 --prompt "a cute kitten"
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


def get_default_device(requested: str) -> str:
    requested = (requested or "auto").lower()
    if requested in {"auto", "best"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/best_evaluated_model",
        help="Path to the fine-tuned Diffusers pipeline folder",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/test_output.png",
        help="Where to save the generated image",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A realistic ginger cat portrait, sharp focus, natural lighting",
        help="Text prompt to generate an image",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, deformed, ugly",
        help="Negative prompt (optional but recommended)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Inference steps (CPU should keep this low: 15-25)",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=7.5,
        help="Guidance scale (CFG), typical range 5.0-10.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        print(f"[ERROR] Model folder not found: {model_dir.resolve()}")
        print("Expected something like: models/best_evaluated_model/")
        return 1

    device = get_default_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    try:
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    except Exception as e:
        print("[ERROR] Missing dependency: diffusers")
        print("Install with: pip install diffusers transformers accelerate safetensors")
        print(f"Details: {e}")
        return 1

    dtype = torch.float16 if device == "cuda" else torch.float32

    print("============================================================")
    print("✅ Text-to-Image Inference Test")
    print(f"Model Dir : {model_dir.resolve()}")
    print(f"Device    : {device}")
    print(f"DType     : {dtype}")
    print(f"Steps     : {args.steps}")
    print(f"CFG       : {args.cfg}")
    print(f"Seed      : {args.seed}")
    print(f"Output    : {out_path.resolve()}")
    print("============================================================")

    # Load pipeline
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        safety_checker=None,  # OK for class project demo; do not use for production safety
        requires_safety_checker=False,
    )

    # Use Euler scheduler by default (often stable + good quality)
    try:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception:
        # If scheduler swap fails, keep whatever is in the model folder
        pass

    pipe = pipe.to(device)

    # Light memory optimizations
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    load_time = time.time() - t0
    print(f"[INFO] Pipeline loaded in {load_time:.2f}s")

    # Generate 1 image
    generator = torch.Generator(device=device).manual_seed(args.seed)
    t1 = time.time()
    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            guidance_scale=float(args.cfg),
            num_inference_steps=int(args.steps),
            generator=generator,
        )
    gen_time = time.time() - t1

    img = result.images[0]
    img.save(str(out_path))

    print("------------------------------------------------------------")
    print(f"✅ Saved: {out_path.resolve()}")
    print(f"⏱️  Generation time: {gen_time:.2f}s")
    print("------------------------------------------------------------")

    # Quick success criteria
    if out_path.exists() and out_path.stat().st_size > 10_000:
        print("✅ TEST PASSED (image created successfully)")
        return 0

    print("❌ TEST FAILED (output file missing or too small)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
