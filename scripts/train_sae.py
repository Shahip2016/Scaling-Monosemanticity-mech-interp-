#!/usr/bin/env python3
"""Train a sparse autoencoder on collected activations."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from scaling_monosemanticity.train import TrainConfig, train_sae
from scaling_monosemanticity.utils import load_activations


def main():
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder")
    parser.add_argument("--activations", default="data/activations.pt")
    parser.add_argument("--n-features", type=int, default=16384)
    parser.add_argument("--d-in", type=int, default=None)
    parser.add_argument("--l1", type=float, default=5.0, help="L1 coefficient (paper uses 5)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--output-dir", default="checkpoints/sae")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    activations = load_activations(args.activations)
    d_in = args.d_in or activations.shape[1]

    config = TrainConfig(
        n_features=args.n_features,
        d_in=d_in,
        l1_coefficient=args.l1,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_steps=args.steps,
        device=device,
        output_dir=args.output_dir,
    )

    print(
        f"Training SAE: {args.n_features:,} features, d_in={d_in}, "
        f"{activations.shape[0]:,} samples, {args.steps} steps"
    )
    sae, history = train_sae(activations, config)
    print(f"Done. Final loss: {history[-1]['loss']:.4f}")
    print(f"Variance explained: {history[-1]['variance_explained']:.3f}")
    print(f"L0 (active features): {history[-1]['l0']:.1f}")


if __name__ == "__main__":
    main()
