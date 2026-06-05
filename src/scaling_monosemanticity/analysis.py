"""Feature interpretability analysis utilities (Section 2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .sae import SparseAutoencoder


@dataclass
class FeatureExample:
    feature_idx: int
    token_idx: int
    activation: float
    context_tokens: list[int]
    highlighted_token: int


def get_top_activating_examples(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    token_ids: torch.Tensor | None,
    feature_idx: int,
    top_k: int = 20,
) -> list[dict]:
    """Return top-k token positions that maximally activate a feature."""
    device = next(sae.parameters()).device
    scaled = sae.feature_activations(activations.to(device)).cpu()

    feature_acts = scaled[:, feature_idx]
    values, indices = torch.topk(feature_acts, k=min(top_k, feature_acts.numel()))

    examples = []
    for act, idx in zip(values.tolist(), indices.tolist()):
        entry = {
            "token_position": idx,
            "activation": act,
        }
        if token_ids is not None:
            window = 32
            start = max(0, idx - window)
            end = min(token_ids.shape[0], idx + window)
            entry["context_token_ids"] = token_ids[start:end].tolist()
            entry["highlight_index"] = idx - start
        examples.append(entry)
    return examples


def analyze_feature(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    feature_idx: int,
    token_ids: torch.Tensor | None = None,
) -> dict:
    """Compute summary statistics for a single feature."""
    device = next(sae.parameters()).device
    with torch.no_grad():
        features = sae.encode_features(activations.to(device)).cpu()
        scaled = sae.feature_activations(activations.to(device)).cpu()

    acts = scaled[:, feature_idx]
    nonzero = acts[acts > 0]

    return {
        "feature_idx": feature_idx,
        "mean_activation": acts.mean().item(),
        "max_activation": acts.max().item(),
        "fraction_active": (acts > 0).float().mean().item(),
        "mean_activation_when_active": nonzero.mean().item() if len(nonzero) else 0.0,
        "top_examples": get_top_activating_examples(
            sae, activations, token_ids, feature_idx
        ),
    }


def find_features_by_keyword(
    sae: SparseAutoencoder,
    model,
    activations: torch.Tensor,
    texts: list[str],
    keyword: str,
    top_k_features: int = 10,
) -> list[dict]:
    """Heuristic search: features that fire most on texts containing a keyword."""
    device = next(sae.parameters()).device
    keyword_lower = keyword.lower()

    matching_indices: list[int] = []
    all_token_ids: list[torch.Tensor] = []

    for text in texts:
        if keyword_lower not in text.lower():
            continue
        ids = model.to_tokens(text, prepend_bos=True)[0]
        all_token_ids.append(ids)

    if not all_token_ids:
        return []

    with torch.no_grad():
        for ids in all_token_ids:
            _, cache = model.run_with_cache(ids.to(device))
            hook_names = [k for k in cache.keys() if "hook_resid_post" in k]
            if not hook_names:
                continue
            resid = cache[hook_names[0]][0]
            from .utils import normalize_activations

            resid = normalize_activations(resid, model.cfg.d_model)
            scaled = sae.feature_activations(resid).mean(dim=0)
            matching_indices.append(scaled)

    if not matching_indices:
        return []

    avg_acts = torch.stack(matching_indices).mean(dim=0)
    values, indices = torch.topk(avg_acts, k=min(top_k_features, avg_acts.numel()))

    return [
        {"feature_idx": int(idx), "mean_activation": float(val)}
        for val, idx in zip(values.tolist(), indices.tolist())
    ]


def score_specificity_rubric(
    activation_strength: float,
    scores: list[int],
) -> dict:
    """Summarize automated interpretability rubric scores (Section 2.1.1).

    Rubric (0-3):
        0 - completely irrelevant
        1 - vaguely related
        2 - loosely related
        3 - cleanly identifies activating text
    """
    if not scores:
        return {"mean_score": 0.0, "histogram": {}}

    hist: dict[int, int] = {}
    for s in scores:
        hist[s] = hist.get(s, 0) + 1

    return {
        "activation_strength": activation_strength,
        "mean_score": sum(scores) / len(scores),
        "histogram": hist,
        "n_samples": len(scores),
    }


def export_feature_dashboard(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    feature_indices: list[int],
    output_dir: str,
    tokenizer=None,
) -> None:
    """Export JSON reports for a set of features."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for idx in feature_indices:
        report = analyze_feature(sae, activations, idx)
        if tokenizer is not None and report["top_examples"]:
            for ex in report["top_examples"]:
                if "context_token_ids" in ex:
                    ex["context_text"] = tokenizer.decode(ex["context_token_ids"])

        with open(out / f"feature_{idx}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
