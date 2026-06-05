"""Scaling laws analysis for SAE hyperparameter selection (Section 1.3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .sae import SparseAutoencoder
from .train import TrainConfig, train_sae


@dataclass
class ScalingSweepConfig:
    d_in: int
    feature_counts: tuple[int, ...] = (4096, 16384, 65536)
    step_counts: tuple[int, ...] = (1000, 2500, 5000)
    learning_rates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    l1_coefficient: float = 5.0
    batch_size: int = 4096
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "scaling_laws"


def estimate_flops(n_features: int, d_in: int, n_steps: int, batch_size: int) -> float:
    """Approximate SAE training FLOPs as proportional to features * steps * batch."""
    forward_flops = 2 * n_features * d_in * batch_size
    return forward_flops * n_steps


def run_scaling_sweep(
    activations: torch.Tensor,
    config: ScalingSweepConfig,
) -> list[dict]:
    """Grid search over features and steps; record loss vs compute."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for n_features in config.feature_counts:
        for n_steps in config.step_counts:
            best_loss = float("inf")
            best_lr = config.learning_rates[0]
            best_history: list[dict] = []

            for lr in config.learning_rates:
                train_config = TrainConfig(
                    n_features=n_features,
                    d_in=config.d_in,
                    l1_coefficient=config.l1_coefficient,
                    learning_rate=lr,
                    batch_size=config.batch_size,
                    n_steps=n_steps,
                    log_every=max(n_steps, 1),
                    save_every=n_steps + 1,
                    device=config.device,
                    output_dir=str(output_dir / f"f{n_features}_s{n_steps}_lr{lr}"),
                )
                _, history = train_sae(activations, train_config)
                final_loss = history[-1]["loss"] if history else float("inf")
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_lr = lr
                    best_history = history

            flops = estimate_flops(n_features, config.d_in, n_steps, config.batch_size)
            results.append(
                {
                    "n_features": n_features,
                    "n_steps": n_steps,
                    "learning_rate": best_lr,
                    "loss": best_loss,
                    "flops": flops,
                    "l0": best_history[-1]["l0"] if best_history else None,
                    "variance_explained": best_history[-1]["variance_explained"]
                    if best_history
                    else None,
                }
            )

    with open(output_dir / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def fit_power_law(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = a * x^b in log space. Returns (a, b)."""
    log_x = np.log(x.astype(np.float64))
    log_y = np.log(y.astype(np.float64))
    b, log_a = np.polyfit(log_x, log_y, 1)
    return float(np.exp(log_a)), float(b)


def compute_optimal_allocation(results: list[dict]) -> dict:
    """For each compute budget, pick the lowest-loss (features, steps) pair."""
    by_flops: dict[float, dict] = {}
    for row in results:
        flops = row["flops"]
        if flops not in by_flops or row["loss"] < by_flops[flops]["loss"]:
            by_flops[flops] = row

    sorted_rows = sorted(by_flops.values(), key=lambda r: r["flops"])
    compute = np.array([r["flops"] for r in sorted_rows])
    loss = np.array([r["loss"] for r in sorted_rows])
    features = np.array([r["n_features"] for r in sorted_rows])
    steps = np.array([r["n_steps"] for r in sorted_rows])

    loss_fit = fit_power_law(compute, loss) if len(compute) >= 2 else (0.0, 0.0)
    feat_fit = fit_power_law(compute, features) if len(compute) >= 2 else (0.0, 0.0)
    step_fit = fit_power_law(compute, steps) if len(compute) >= 2 else (0.0, 0.0)

    return {
        "compute_optimal_runs": sorted_rows,
        "loss_power_law": {"a": loss_fit[0], "b": loss_fit[1]},
        "features_power_law": {"a": feat_fit[0], "b": feat_fit[1]},
        "steps_power_law": {"a": step_fit[0], "b": step_fit[1]},
    }


def recommend_hyperparameters(
    target_flops: float,
    allocation: dict,
) -> dict:
    """Extrapolate compute-optimal features, steps, and learning rate."""
    feat_a = allocation["features_power_law"]["a"]
    feat_b = allocation["features_power_law"]["b"]
    step_a = allocation["steps_power_law"]["a"]
    step_b = allocation["steps_power_law"]["b"]

    return {
        "target_flops": target_flops,
        "recommended_n_features": int(feat_a * (target_flops**feat_b)),
        "recommended_n_steps": int(step_a * (target_flops**step_b)),
        "note": "Extrapolated from sweep; validate with a short pilot run.",
    }


def plot_scaling_laws(results: list[dict], output_path: str) -> None:
    """Save scaling law plots if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    allocation = compute_optimal_allocation(results)
    runs = allocation["compute_optimal_runs"]
    compute = [r["flops"] for r in runs]
    loss = [r["loss"] for r in runs]
    features = [r["n_features"] for r in runs]
    steps = [r["n_steps"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].loglog(compute, loss, "o-")
    axes[0].set_xlabel("Compute (FLOPs)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss vs Compute")

    axes[1].loglog(compute, features, "o-")
    axes[1].set_xlabel("Compute (FLOPs)")
    axes[1].set_ylabel("Optimal # Features")
    axes[1].set_title("Features vs Compute")

    axes[2].loglog(compute, steps, "o-")
    axes[2].set_xlabel("Compute (FLOPs)")
    axes[2].set_ylabel("Optimal # Steps")
    axes[2].set_title("Steps vs Compute")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
