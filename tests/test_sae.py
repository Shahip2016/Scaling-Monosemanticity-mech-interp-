"""Unit tests for sparse autoencoder."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scaling_monosemanticity.utils import normalize_activations
from scaling_monosemanticity.sae import SAEConfig, SparseAutoencoder


def test_sae_forward_and_loss():
    d_in, n_features, batch = 64, 256, 32
    config = SAEConfig(d_in=d_in, n_features=n_features, l1_coefficient=5.0)
    sae = SparseAutoencoder(config)
    x = torch.randn(batch, d_in)

    recon, features = sae(x)
    assert recon.shape == x.shape
    assert features.shape == (batch, n_features)
    assert (features >= 0).all()

    loss = sae.compute_loss(x)
    assert loss.total.ndim == 0
    assert loss.l0.item() <= n_features


def test_normalization():
    d_model = 128
    x = torch.randn(100, d_model)
    normed = normalize_activations(x, d_model)
    avg_sq_norm = (normed**2).sum(dim=-1).mean()
    assert torch.isclose(avg_sq_norm, torch.tensor(float(d_model)), rtol=0.01)


def test_feature_activations_scaled():
    d_in, n_features = 32, 64
    sae = SparseAutoencoder(SAEConfig(d_in=d_in, n_features=n_features))
    x = torch.randn(8, d_in)
    scaled = sae.feature_activations(x)
    raw = sae.encode_features(x)
    norms = sae.W_dec.norm(dim=0)
    assert torch.allclose(scaled, raw * norms, atol=1e-5)


def test_save_load():
    config = SAEConfig(d_in=32, n_features=64)
    sae = SparseAutoencoder(config)
    path = Path(__file__).parent / "_test_sae.pt"
    sae.save(str(path))
    loaded = SparseAutoencoder.load(str(path))
    x = torch.randn(4, 32)
    assert torch.allclose(sae(x)[0], loaded(x)[0], atol=1e-5)
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    test_sae_forward_and_loss()
    test_normalization()
    test_feature_activations_scaled()
    test_save_load()
    print("All tests passed.")
