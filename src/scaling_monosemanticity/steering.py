"""Feature steering: causal interventions on model behavior (Section 2.1)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

from .sae import SparseAutoencoder


@dataclass
class SteeringConfig:
    feature_idx: int
    coefficient: float = 10.0
    hook_layer: int | None = None


def _get_hook_layer(model: HookedTransformer, hook_layer: int | None) -> int:
    return hook_layer if hook_layer is not None else model.cfg.n_layers // 2


def make_steering_hook(
    sae: SparseAutoencoder,
    config: SteeringConfig,
    d_model: int,
):
    """Create a forward hook that adds a feature direction to the residual stream."""
    device = next(sae.parameters()).device
    decoder_col = sae.W_dec[:, config.feature_idx].detach().to(device)
    delta = config.coefficient * decoder_col

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        activation = activation.clone()
        activation += delta
        return activation

    return hook_fn


@torch.no_grad()
def generate_with_steering(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    prompt: str,
    config: SteeringConfig,
    max_new_tokens: int = 50,
) -> str:
    """Generate text with a feature steering intervention."""
    layer = _get_hook_layer(model, config.hook_layer)
    hook_name = f"blocks.{layer}.hook_resid_post"
    hook_fn = make_steering_hook(sae, config, model.cfg.d_model)

    tokens = model.to_tokens(prompt)
    generated = model.generate(
        tokens,
        max_new_tokens=max_new_tokens,
        fwd_hooks=[(hook_name, hook_fn)],
    )
    return model.to_string(generated[0])


@torch.no_grad()
def compare_steering(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    prompt: str,
    feature_idx: int,
    coefficients: list[float] | None = None,
    max_new_tokens: int = 40,
) -> dict[str, str]:
    """Generate baseline and steered outputs for comparison."""
    coefficients = coefficients or [-20.0, 0.0, 20.0]
    results = {}

    tokens = model.to_tokens(prompt)
    baseline = model.generate(tokens, max_new_tokens=max_new_tokens)
    results["baseline"] = model.to_string(baseline[0])

    for coef in coefficients:
        if coef == 0.0:
            continue
        config = SteeringConfig(feature_idx=feature_idx, coefficient=coef)
        results[f"coef_{coef}"] = generate_with_steering(
            model, sae, prompt, config, max_new_tokens
        )

    return results


@torch.no_grad()
def measure_feature_effect_on_logits(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    prompt: str,
    feature_idx: int,
    coefficient: float,
    hook_layer: int | None = None,
) -> dict:
    """Measure how steering shifts next-token logit distribution."""
    layer = _get_hook_layer(model, hook_layer)
    hook_name = f"blocks.{layer}.hook_resid_post"
    tokens = model.to_tokens(prompt)

    baseline_logits = model(tokens)
    baseline_probs = torch.softmax(baseline_logits[0, -1], dim=-1)

    config = SteeringConfig(
        feature_idx=feature_idx,
        coefficient=coefficient,
        hook_layer=layer,
    )
    hook_fn = make_steering_hook(sae, config, model.cfg.d_model)
    steered_logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    steered_probs = torch.softmax(steered_logits[0, -1], dim=-1)

    diff = steered_probs - baseline_probs
    top_increase_idx = diff.argmax().item()
    top_decrease_idx = diff.argmin().item()

    return {
        "top_increase_token": model.to_string(top_increase_idx),
        "top_increase_delta": diff[top_increase_idx].item(),
        "top_decrease_token": model.to_string(top_decrease_idx),
        "top_decrease_delta": diff[top_decrease_idx].item(),
        "kl_divergence": torch.nn.functional.kl_div(
            steered_probs.log().clamp(min=-1e8),
            baseline_probs,
            reduction="sum",
        ).item(),
    }
