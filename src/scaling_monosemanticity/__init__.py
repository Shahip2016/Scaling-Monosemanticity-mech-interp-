"""Scaling Monosemanticity: Sparse Autoencoder implementation for transformer interpretability."""

__version__ = "0.1.0"

from .sae import SparseAutoencoder, SAEConfig

__all__ = [
    "SparseAutoencoder",
    "SAEConfig",
]


def __getattr__(name: str):
    if name in ("ActivationBuffer", "collect_activations"):
        from .activation_buffer import ActivationBuffer, collect_activations

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
