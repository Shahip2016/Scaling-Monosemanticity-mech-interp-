"""Training loop for sparse autoencoders."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .sae import SAEConfig, SparseAutoencoder


@dataclass
class TrainConfig:
    n_features: int
    d_in: int
    l1_coefficient: float = 5.0
    learning_rate: float = 3e-4
    batch_size: int = 4096
    n_steps: int = 10_000
    log_every: int = 100
    save_every: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    output_dir: str = "checkpoints"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloader(
    activations: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TensorDataset(activations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def train_sae(
    activations: torch.Tensor,
    config: TrainConfig,
    resume_path: str | None = None,
) -> tuple[SparseAutoencoder, list[dict]]:
    """Train an SAE on pre-collected activations for one epoch over the data."""
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae_config = SAEConfig(
        d_in=config.d_in,
        n_features=config.n_features,
        l1_coefficient=config.l1_coefficient,
    )
    sae = SparseAutoencoder(sae_config).to(config.device)

    if resume_path:
        sae = SparseAutoencoder.load(resume_path, device=config.device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.learning_rate)
    dataloader = make_dataloader(activations, config.batch_size)
    data_iter = iter(dataloader)

    history: list[dict] = []
    start = time.time()

    for step in range(1, config.n_steps + 1):
        try:
            batch = next(data_iter)[0]
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)[0]

        batch = batch.to(config.device)
        loss_out = sae.compute_loss(batch)

        optimizer.zero_grad()
        loss_out.total.backward()
        optimizer.step()

        if step % config.log_every == 0 or step == 1:
            record = {
                "step": step,
                "loss": loss_out.total.item(),
                "mse": loss_out.mse.item(),
                "l1": loss_out.l1.item(),
                "l0": loss_out.l0.item(),
                "variance_explained": loss_out.variance_explained.item(),
                "elapsed_s": time.time() - start,
            }
            history.append(record)
            print(
                f"step {step:6d} | loss {record['loss']:.4f} | "
                f"mse {record['mse']:.4f} | l0 {record['l0']:.1f} | "
                f"var {record['variance_explained']:.3f}"
            )

        if step % config.save_every == 0:
            sae.save(str(output_dir / f"sae_step_{step}.pt"))

    final_path = output_dir / "sae_final.pt"
    sae.save(str(final_path))

    dead_mask = sae.get_dead_features_mask(activations)
    dead_frac = dead_mask.float().mean().item()

    summary = {
        "config": asdict(config),
        "final_loss": history[-1] if history else {},
        "dead_feature_fraction": dead_frac,
        "n_dead_features": int(dead_mask.sum().item()),
        "checkpoint": str(final_path),
    }
    with open(output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return sae, history
