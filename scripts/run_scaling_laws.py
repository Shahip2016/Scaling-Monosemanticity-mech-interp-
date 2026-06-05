#!/usr/bin/env python3
"""Run scaling laws hyperparameter sweep (Section 1.3)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scaling_monosemanticity.scaling_laws import (
    ScalingSweepConfig,
    compute_optimal_allocation,
    plot_scaling_laws,
    recommend_hyperparameters,
    run_scaling_sweep,
)
from scaling_monosemanticity.utils import load_activations


def main():
    parser = argparse.ArgumentParser(description="SAE scaling laws sweep")
    parser.add_argument("--activations", default="data/activations.pt")
    parser.add_argument("--features", nargs="+", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--steps", nargs="+", type=int, default=[1000, 2500, 5000])
    parser.add_argument("--output-dir", default="results/scaling_laws")
    parser.add_argument("--target-flops", type=float, default=1e15)
    args = parser.parse_args()

    activations = load_activations(args.activations)
    d_in = activations.shape[1]

    config = ScalingSweepConfig(
        d_in=d_in,
        feature_counts=tuple(args.features),
        step_counts=tuple(args.steps),
        output_dir=args.output_dir,
    )

    print("Running scaling laws sweep...")
    results = run_scaling_sweep(activations, config)
    allocation = compute_optimal_allocation(results)
    recommendation = recommend_hyperparameters(args.target_flops, allocation)

    with open(Path(args.output_dir) / "allocation.json", "w", encoding="utf-8") as f:
        json.dump({"allocation": allocation, "recommendation": recommendation}, f, indent=2)

    plot_scaling_laws(results, str(Path(args.output_dir) / "scaling_laws.png"))
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
