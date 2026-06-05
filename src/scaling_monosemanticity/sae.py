"""Sparse Autoencoder architecture from Scaling Monosemanticity (Anthropic, 2024).

Reference: https://arxiv.org/abs/2605.29358
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    """Configuration for a sparse autoencoder."""

    d_in: int
    n_features: int
    l1_coefficient: float = 5.0
    tied_weights: bool = False

    @property
    def expansion_factor(self) -> float:
        return self.n_features / self.d_in


class SAELoss(NamedTuple):
    """Loss components returned by SparseAutoencoder.compute_loss."""

    total: torch.Tensor
    mse: torch.Tensor
    l1: torch.Tensor
    l0: torch.Tensor
    variance_explained: torch.Tensor


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for decomposing transformer residual stream activations.

    Architecture (Section 1.1):
        encoder: x -> ReLU(W_enc @ x + b_enc)
        decoder: sum_i f_i(x) * W_dec[:, i] + b_dec

    Loss:
        L = ||x - x_hat||^2 + lambda * sum_i f_i(x) * ||W_dec[:, i]||_2
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        d_in, n_features = config.d_in, config.n_features

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(n_features, d_in))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(d_in, n_features))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self.W_dec.data = self._normalize_decoder_columns(self.W_dec.data)

    @staticmethod
    def _normalize_decoder_columns(w_dec: torch.Tensor) -> torch.Tensor:
        norms = w_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        return w_dec / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-activation encoder output (before ReLU)."""
        return F.linear(x, self.W_enc, self.b_enc)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return sparse feature activations f_i(x) = ReLU(encoder output)."""
        return F.relu(self.encode(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Reconstruct activations from feature activations."""
        return F.linear(features, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_features(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Scaled feature activations: f_i(x) * ||W_dec[:, i]||_2."""
        features = self.encode_features(x)
        decoder_norms = self.W_dec.norm(dim=0)
        return features * decoder_norms

    def compute_loss(self, x: torch.Tensor) -> SAELoss:
        reconstruction, features = self.forward(x)
        decoder_norms = self.W_dec.norm(dim=0)

        mse = F.mse_loss(reconstruction, x)
        l1 = (features * decoder_norms).sum(dim=-1).mean()
        total = mse + self.config.l1_coefficient * l1

        l0 = (features > 0).float().sum(dim=-1).mean()

        x_centered = x - x.mean(dim=0, keepdim=True)
        var = (x_centered**2).sum(dim=-1).mean().clamp(min=1e-8)
        residual = x - reconstruction
        unexplained = (residual**2).sum(dim=-1).mean()
        variance_explained = 1.0 - unexplained / var

        return SAELoss(
            total=total,
            mse=mse,
            l1=l1,
            l0=l0,
            variance_explained=variance_explained,
        )

    @torch.no_grad()
    def get_dead_features_mask(
        self,
        activations: torch.Tensor,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """Mark features that never fired on the provided activation sample."""
        device = next(self.parameters()).device
        n_features = self.config.n_features
        fired = torch.zeros(n_features, dtype=torch.bool, device=device)

        for start in range(0, activations.shape[0], batch_size):
            batch = activations[start : start + batch_size].to(device)
            features = self.encode_features(batch)
            fired |= (features > 0).any(dim=0)

        return ~fired

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | torch.device = "cpu") -> SparseAutoencoder:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        sae = cls(checkpoint["config"])
        sae.load_state_dict(checkpoint["state_dict"])
        return sae.to(device)
