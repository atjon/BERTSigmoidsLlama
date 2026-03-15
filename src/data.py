"""
data.py — Load and preprocess the Anthropic HH-RLHF dataset.

Two public functions:

load_pair_datasets() — for training with pairwise ranking loss.
  1. Download helpful-base and harmless-base subsets via HuggingFace `datasets`.
  2. Sample HELPFUL_TRAIN_N + HARMLESS_TRAIN_N preference pairs.
  3. Combine and split 80/20 at the **pair level** (before flattening) to avoid
     correlated samples across train/val.
  4. Tokenize chosen and rejected separately; truncate from the LEFT to preserve
     the final assistant response at the end of each conversation.
  5. Return (train_pair_ds, val_pair_ds, test_flat_ds).
     - train_pair_ds / val_pair_ds have fields:
         chosen_input_ids, chosen_attention_mask,
         rejected_input_ids, rejected_attention_mask
     - test_flat_ds (for the interpretability pipeline) has fields:
         input_ids, attention_mask, label, text

load_datasets() — legacy flat format, kept for compatibility with the
  interpretability pipeline when called standalone.
"""

from __future__ import annotations

import logging
from typing import Tuple

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import BertTokenizer

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_raw_pairs(subset_name: str, split: str, n: int) -> Dataset:
    """Load one HH-RLHF subset split and cap at n pairs. Does NOT flatten."""
    raw = load_dataset("Anthropic/hh-rlhf", data_dir=subset_name, split=split)
    actual = len(raw)
    if actual < n:
        logger.warning(
            "%s/%s has only %d rows (requested %d); using all available.",
            subset_name, split, actual, n,
        )
        n = actual
    return raw.shuffle(seed=config.SEED).select(range(n))


def _build_pair_dataset(raw_pairs: Dataset, tokenizer: BertTokenizer) -> Dataset:
    """
    Tokenize chosen and rejected texts independently, keeping them paired.

    Returns a dataset with fields:
        chosen_input_ids, chosen_attention_mask,
        rejected_input_ids, rejected_attention_mask
    """
    tokenizer.truncation_side = "left"

    def _process(batch):
        chosen_enc = tokenizer(
            batch["chosen"],
            max_length=config.MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        rejected_enc = tokenizer(
            batch["rejected"],
            max_length=config.MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        return {
            "chosen_input_ids":       chosen_enc["input_ids"],
            "chosen_attention_mask":  chosen_enc["attention_mask"],
            "rejected_input_ids":     rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc["attention_mask"],
        }

    return raw_pairs.map(
        _process,
        batched=True,
        remove_columns=raw_pairs.column_names,
        desc="Tokenising pairs",
    )


def _flatten_pairs(ds: Dataset, tokenizer: BertTokenizer) -> Dataset:
    """
    Convert a dataset of (chosen, rejected) pairs into individual labelled rows.

    Truncates from the LEFT so the final assistant response is preserved.
    Each output row has: input_ids, attention_mask, label (int), text (str).
    """
    tokenizer.truncation_side = "left"

    def _process(batch):
        texts  = batch["chosen"] + batch["rejected"]
        labels = [1] * len(batch["chosen"]) + [0] * len(batch["rejected"])

        enc = tokenizer(
            texts,
            max_length=config.MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label":          labels,
            "text":           texts,
        }

    return ds.map(
        _process,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenising (flat)",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pair_datasets(
    tokenizer: BertTokenizer | None = None,
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Build train/val pair datasets, a pair-format test set, and a flat test set.

    Train/val are split at the pair level before flattening so that chosen_A
    and rejected_A always end up in the same split.

    Args:
        tokenizer: Optional pre-loaded BertTokenizer. Loads bert-base-uncased if None.

    Returns:
        (train_pair_ds, val_pair_ds, test_pair_ds, test_flat_ds)
        - train_pair_ds / val_pair_ds / test_pair_ds: paired format for ranking loss
          and pairwise accuracy evaluation (score_chosen > score_rejected)
        - test_flat_ds: flat individual-row format for the interpretability pipeline
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_ID)

    logger.info("Loading HH-RLHF — helpful-base …")
    helpful_train_raw = _sample_raw_pairs("helpful-base", "train", config.HELPFUL_TRAIN_N)
    helpful_test_raw  = _sample_raw_pairs("helpful-base", "test",  config.TEST_N_PER_SUBSET)

    logger.info("Loading HH-RLHF — harmless-base …")
    harmless_train_raw = _sample_raw_pairs("harmless-base", "train", config.HARMLESS_TRAIN_N)
    harmless_test_raw  = _sample_raw_pairs("harmless-base", "test",  config.TEST_N_PER_SUBSET)

    # --- Pair-level train/val split (before any flattening) ---
    combined_raw = concatenate_datasets([helpful_train_raw, harmless_train_raw])
    combined_raw = combined_raw.shuffle(seed=config.SEED)
    pair_split   = combined_raw.train_test_split(test_size=config.VAL_SPLIT, seed=config.SEED)

    train_pair_ds = _build_pair_dataset(pair_split["train"], tokenizer)
    val_pair_ds   = _build_pair_dataset(pair_split["test"],  tokenizer)

    # --- Test sets ---
    test_raw      = concatenate_datasets([helpful_test_raw, harmless_test_raw]).shuffle(seed=config.SEED)
    test_pair_ds  = _build_pair_dataset(test_raw, tokenizer)   # for pairwise eval
    test_flat     = _flatten_pairs(test_raw, tokenizer)         # for interpretability pipeline

    # Set torch format
    pair_cols = ["chosen_input_ids", "chosen_attention_mask",
                 "rejected_input_ids", "rejected_attention_mask"]
    flat_cols  = ["input_ids", "attention_mask", "label"]

    train_pair_ds = train_pair_ds.with_format("torch", columns=pair_cols)
    val_pair_ds   = val_pair_ds.with_format("torch",   columns=pair_cols)
    test_pair_ds  = test_pair_ds.with_format("torch",  columns=pair_cols)
    test_flat     = test_flat.with_format("torch", columns=flat_cols, output_all_columns=True)

    logger.info(
        "Dataset sizes — train pairs: %d  val pairs: %d  test pairs: %d  test rows: %d",
        len(train_pair_ds), len(val_pair_ds), len(test_pair_ds), len(test_flat),
    )
    return train_pair_ds, val_pair_ds, test_pair_ds, test_flat


def load_datasets(
    tokenizer: BertTokenizer | None = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Legacy flat-format loader (individual rows, binary labels).
    Kept for standalone interpretability use. Prefer load_pair_datasets() for training.

    Returns (train_dataset, val_dataset, test_dataset) in flat format.
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_ID)

    logger.info("Loading HH-RLHF — helpful-base …")
    helpful_train_raw = _sample_raw_pairs("helpful-base", "train", config.HELPFUL_TRAIN_N)
    helpful_test_raw  = _sample_raw_pairs("helpful-base", "test",  config.TEST_N_PER_SUBSET)

    logger.info("Loading HH-RLHF — harmless-base …")
    harmless_train_raw = _sample_raw_pairs("harmless-base", "train", config.HARMLESS_TRAIN_N)
    harmless_test_raw  = _sample_raw_pairs("harmless-base", "test",  config.TEST_N_PER_SUBSET)

    helpful_train  = _flatten_pairs(helpful_train_raw,  tokenizer)
    harmless_train = _flatten_pairs(harmless_train_raw, tokenizer)

    combined_train = concatenate_datasets([helpful_train, harmless_train])
    combined_train = combined_train.shuffle(seed=config.SEED)
    split    = combined_train.train_test_split(test_size=config.VAL_SPLIT, seed=config.SEED)
    train_ds = split["train"]
    val_ds   = split["test"]

    test_raw = concatenate_datasets([helpful_test_raw, harmless_test_raw]).shuffle(seed=config.SEED)
    test_ds  = _flatten_pairs(test_raw, tokenizer)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.with_format("torch", columns=cols, output_all_columns=True)
    val_ds   = val_ds.with_format("torch",   columns=cols, output_all_columns=True)
    test_ds  = test_ds.with_format("torch",  columns=cols, output_all_columns=True)

    logger.info(
        "Dataset sizes — train: %d  val: %d  test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds
