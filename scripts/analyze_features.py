#!/usr/bin/env python3
"""Analyze and export interpretable SAE features."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformer_lens import HookedTransformer

from scaling_monosemanticity.analysis import export_feature_dashboard, find_features_by_keyword
from scaling_monosemanticity.sae import SparseAutoencoder
from scaling_monosemanticity.utils import load_activations


SAMPLE_TEXTS = [
    "The Golden Gate Bridge is an iconic suspension bridge in San Francisco.",
    "I really enjoy books on neuroscience that change how I think about perception.",
    "Every train line has to cross one particular bridge, which is a massive chokepoint.",
    "You have to go to the big tourist attractions like the Louvre and the Eiffel Tower.",
    "The pyramids in Egypt are among the most interesting things to visit.",
    "BART runs through the Transbay Tube which requires a lot of attention.",
]


def main():
    parser = argparse.ArgumentParser(description="Analyze SAE features")
    parser.add_argument("--sae", default="checkpoints/sae/sae_final.pt")
    parser.add_argument("--activations", default="data/activations.pt")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--keyword", default=None, help="Search features by keyword")
    parser.add_argument("--features", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default="results/features")
    args = parser.parse_args()

    sae = SparseAutoencoder.load(args.sae)
    activations = load_activations(args.activations)
    model = HookedTransformer.from_pretrained(args.model)

    if args.keyword:
        matches = find_features_by_keyword(sae, model, activations, SAMPLE_TEXTS, args.keyword)
        print(f"Top features for '{args.keyword}':")
        for m in matches:
            print(f"  feature {m['feature_idx']}: activation {m['mean_activation']:.4f}")
        feature_ids = [m["feature_idx"] for m in matches]
    else:
        feature_ids = args.features or [0, 1, 2, 3]

    export_feature_dashboard(
        sae,
        activations,
        feature_ids,
        args.output_dir,
        tokenizer=model.tokenizer,
    )
    print(f"Exported {len(feature_ids)} feature reports to {args.output_dir}")


if __name__ == "__main__":
    main()
