"""Collect and normalize residual stream activations from a transformer."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformer_lens import HookedTransformer


@dataclass
class ActivationBufferConfig:
    model_name: str = "gpt2"
    hook_layer: int | None = None
    dataset_name: str = "Skylion007/openwebtext"
    dataset_split: str = "train"
    max_tokens: int = 1_000_000
    batch_size: int = 8
    seq_len: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


def get_middle_layer(model: HookedTransformer) -> int:
    """Return the middle transformer block index."""
    return model.cfg.n_layers // 2


from .utils import normalize_activations


class _TokenStreamDataset(IterableDataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __iter__(self) -> Iterator[torch.Tensor]:
        for start in range(0, len(self.token_ids) - self.seq_len, self.seq_len):
            chunk = self.token_ids[start : start + self.seq_len]
            yield torch.tensor(chunk, dtype=torch.long)


class ActivationBuffer:
    """Stream normalized residual activations from a hook point."""

    def __init__(
        self,
        model: HookedTransformer,
        hook_name: str,
        d_model: int,
        device: str,
    ):
        self.model = model
        self.hook_name = hook_name
        self.d_model = d_model
        self.device = device
        self._cache: list[torch.Tensor] = []

    def _capture_hook(self, activation: torch.Tensor, hook) -> torch.Tensor:
        flat = activation.reshape(-1, activation.shape[-1])
        self._cache.append(flat.detach().cpu())
        return activation

    @torch.no_grad()
    def collect_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run model on tokens and return normalized residual activations."""
        self._cache = []
        tokens = tokens.to(self.device)
        self.model.run_with_hooks(
            tokens,
            fwd_hooks=[(self.hook_name, self._capture_hook)],
        )
        activations = torch.cat(self._cache, dim=0)
        return normalize_activations(activations, self.d_model)

    @torch.no_grad()
    def iter_batches(
        self,
        dataloader: DataLoader,
        max_batches: int | None = None,
    ) -> Iterator[torch.Tensor]:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            yield self.collect_from_tokens(batch)


def load_model_and_hook(
    model_name: str,
    hook_layer: int | None,
    device: str,
) -> tuple[HookedTransformer, str, int]:
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        center_writing_weights=False,
    )
    layer = hook_layer if hook_layer is not None else get_middle_layer(model)
    hook_name = f"blocks.{layer}.hook_resid_post"
    return model, hook_name, layer


def tokenize_dataset(config: ActivationBufferConfig, tokenizer) -> list[int]:
    dataset = load_dataset(
        config.dataset_name,
        split=config.dataset_split,
        streaming=True,
        trust_remote_code=True,
    )
    token_ids: list[int] = []
    for example in dataset:
        text = example.get("text") or example.get("content") or str(example)
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)
        if len(token_ids) >= config.max_tokens:
            break
    return token_ids[: config.max_tokens]


def collect_activations(
    config: ActivationBufferConfig,
    save_path: str | None = None,
) -> torch.Tensor:
    """Collect activations from the model and optionally persist to disk."""
    model, hook_name, _ = load_model_and_hook(
        config.model_name,
        config.hook_layer,
        config.device,
    )
    token_ids = tokenize_dataset(config, model.tokenizer)
    dataset = _TokenStreamDataset(token_ids, config.seq_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    buffer = ActivationBuffer(
        model=model,
        hook_name=hook_name,
        d_model=model.cfg.d_model,
        device=config.device,
    )

    chunks: list[torch.Tensor] = []
    for batch in buffer.iter_batches(dataloader):
        chunks.append(batch)

    activations = torch.cat(chunks, dim=0)
    if save_path:
        torch.save(
            {
                "activations": activations,
                "hook_name": hook_name,
                "model_name": config.model_name,
                "d_model": model.cfg.d_model,
            },
            save_path,
        )
    return activations
