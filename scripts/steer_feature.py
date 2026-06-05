#!/usr/bin/env python3
"""Steer model behavior using an SAE feature direction."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformer_lens import HookedTransformer

from scaling_monosemanticity.sae import SparseAutoencoder
from scaling_monosemanticity.steering import compare_steering, measure_feature_effect_on_logits


def main():
    parser = argparse.ArgumentParser(description="Steer model with SAE feature")
    parser.add_argument("--sae", default="checkpoints/sae/sae_final.pt")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--feature", type=int, required=True)
    parser.add_argument("--prompt", default="The famous bridge in San Francisco is the")
    parser.add_argument("--coefficients", nargs="+", type=float, default=[-30, 30])
    parser.add_argument("--max-tokens", type=int, default=40)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sae = SparseAutoencoder.load(args.sae)
    model = HookedTransformer.from_pretrained(args.model)

    results = compare_steering(
        model,
        sae,
        args.prompt,
        args.feature,
        coefficients=[0.0] + args.coefficients,
        max_new_tokens=args.max_tokens,
    )

    logit_effect = measure_feature_effect_on_logits(
        model, sae, args.prompt, args.feature, coefficient=args.coefficients[-1]
    )

    output = {"generations": results, "logit_effect": logit_effect}
    print(json.dumps(output, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
