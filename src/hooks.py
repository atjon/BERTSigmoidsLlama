"""
hooks.py — PyTorch forward hook utilities for capturing BertIntermediate activations.

The hook target is:
    model.bert.encoder.layer[LAYER].intermediate
    (BertForSequenceClassification wraps the encoder under model.bert)

This differs from the proof-of-concept notebook where the bare BertModel was
hooked at model.encoder.layer[LAYER].intermediate.

BertIntermediate applies: linear(768 → 3072) + GELU.
Its output shape is (batch, seq_len, 3072).
We mean-pool over the sequence dimension → (batch, 3072).

Public API:
    hooked_model(model, layer_idx)          — context manager; yields ActivationCapture
    capture_activations(model, dataloader, layer_idx, device)
                                            — runs inference; returns (activations, labels, texts)
    save_activations(path, activations, labels, texts)
    load_activations(path)                  → (activations, labels, texts)
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation accumulator
# ---------------------------------------------------------------------------

class ActivationCapture:
    """Accumulates mean-pooled FFN activations across batches."""

    def __init__(self):
        self._batches: List[torch.Tensor] = []

    def hook_fn(self, module, input, output):
        # output: (batch, seq_len, 3072) — mean-pool over tokens → (batch, 3072)
        self._batches.append(output.detach().mean(dim=1).cpu())

    def get_all(self) -> torch.Tensor:
        """Return all accumulated activations as a single (N, 3072) tensor."""
        return torch.cat(self._batches, dim=0)

    def clear(self):
        self._batches = []


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@contextmanager
def hooked_model(model: BertForSequenceClassification, layer_idx: int):
    """
    Register a forward hook on BertIntermediate at layer_idx, yield the
    ActivationCapture object, then remove the hook on exit (even on error).

    Usage:
        with hooked_model(model, config.INTERP_LAYER) as cap:
            model(input_ids=..., attention_mask=...)
        acts = cap.get_all()
    """
    capture = ActivationCapture()
    target  = model.bert.encoder.layer[layer_idx].intermediate
    handle  = target.register_forward_hook(capture.hook_fn)
    try:
        yield capture
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Batch inference with activation capture
# ---------------------------------------------------------------------------

def capture_activations(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    layer_idx: int,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Run full inference over dataloader with a hook on layer_idx.

    Returns:
        activations: (N, 3072) float tensor — mean-pooled FFN activations
        labels:      (N,) long tensor
        texts:       List[str] of raw input texts (length N)
    """
    if device is None:
        device = config.get_device()

    model.eval()
    all_labels: List[torch.Tensor] = []
    all_texts:  List[str]          = []

    with hooked_model(model, layer_idx) as capture:
        with torch.no_grad():
            for batch in dataloader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                model(input_ids=input_ids, attention_mask=attention_mask)

                all_labels.append(batch["label"])
                # 'text' may be a list of strings or a batched tensor of strings;
                # HuggingFace datasets with output_all_columns=True returns strings.
                texts = batch.get("text", [""] * len(batch["label"]))
                if isinstance(texts, torch.Tensor):
                    texts = [t for t in texts]
                all_texts.extend(texts)

    activations = capture.get_all()              # (N, 3072)
    labels      = torch.cat(all_labels, dim=0)   # (N,)

    logger.info(
        "Captured activations: shape=%s  labels=%s",
        tuple(activations.shape), tuple(labels.shape),
    )
    return activations, labels, all_texts


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_activations(
    path: str,
    activations: torch.Tensor,
    labels: torch.Tensor,
    texts: List[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"activations": activations, "labels": labels, "texts": texts}, path)
    logger.info("Activations saved to %s", path)


def load_activations(path: str) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    data = torch.load(path, map_location="cpu")
    logger.info("Activations loaded from %s", path)
    return data["activations"], data["labels"], data["texts"]
