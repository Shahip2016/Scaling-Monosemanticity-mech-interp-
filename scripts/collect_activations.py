#!/usr/bin/env python3
"""Collect residual stream activations from a transformer."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scaling_monosemanticity.activation_buffer import (
    ActivationBufferConfig,
    collect_activations,
)


def main():
    parser = argparse.ArgumentParser(description="Collect model activations for SAE training")
    parser.add_argument("--model", default="gpt2", help="HuggingFace / TransformerLens model name")
    parser.add_argument("--layer", type=int, default=None, help="Hook layer (default: middle)")
    parser.add_argument("--max-tokens", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output", default="data/activations.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = ActivationBufferConfig(
        model_name=args.model,
        hook_layer=args.layer,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
    )

    print(f"Collecting activations from {args.model} on {device}...")
    activations = collect_activations(config, save_path=args.output)
    print(f"Saved {activations.shape[0]:,} activation vectors to {args.output}")


if __name__ == "__main__":
    main()
