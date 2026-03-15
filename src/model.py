"""
model.py — BERT binary preference classifier.

Uses BertForSequenceClassification with num_labels=2, which attaches a single
linear layer on top of the [CLS] token representation.  HuggingFace initialises
the new classification head automatically; the BERT encoder weights are loaded
from the pretrained checkpoint.
"""

from __future__ import annotations

import os
import logging

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from src import config

logger = logging.getLogger(__name__)


def build_model(device: torch.device | None = None) -> BertForSequenceClassification:
    """
    Load bert-base-uncased with a 2-class classification head.

    Args:
        device: Target device. Defaults to config.get_device().

    Returns:
        Model moved to device and set to train mode.
    """
    if device is None:
        device = config.get_device()

    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_ID,
        num_labels=1,   # scalar reward score; pairwise ranking loss computed in train.py
    )
    model.to(device)
    logger.info("Loaded %s on %s", config.MODEL_ID, device)
    return model


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> BertForSequenceClassification:
    """
    Load a fine-tuned model from a saved checkpoint directory.

    Args:
        checkpoint_path: Path to a directory produced by model.save_pretrained().
        device: Target device. Defaults to config.get_device().

    Returns:
        Model in eval mode on device.
    """
    if device is None:
        device = config.get_device()

    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from %s on %s", checkpoint_path, device)
    return model


def best_checkpoint_path() -> str | None:
    """
    Return the path to the latest epoch checkpoint, or None if none exist.
    Epochs are saved as checkpoints/epoch-0/, epoch-1/, etc.
    """
    if not os.path.isdir(config.CHECKPOINT_DIR):
        return None
    epochs = sorted(
        [
            d for d in os.listdir(config.CHECKPOINT_DIR)
            if d.startswith("epoch-") and os.path.isdir(
                os.path.join(config.CHECKPOINT_DIR, d)
            )
        ]
    )
    if not epochs:
        return None
    return os.path.join(config.CHECKPOINT_DIR, epochs[-1])
