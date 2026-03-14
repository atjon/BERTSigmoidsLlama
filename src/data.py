"""
data.py — Load and preprocess the Anthropic HH-RLHF dataset.

Pipeline:
  1. Download helpful-base and harmless-base subsets via HuggingFace `datasets`.
  2. Sample HELPFUL_TRAIN_N + HARMLESS_TRAIN_N preference pairs for training.
  3. Flatten each pair into two rows: chosen (label=1) and rejected (label=0).
  4. Tokenize with bert-base-uncased; truncate to MAX_SEQ_LEN from the right,
     which preserves the "Human:" / "Assistant:" markers at the string start.
  5. Apply 80/20 train/val split over the combined training rows.
  6. Build a separate test set from each subset's own test partition
     (TEST_N_PER_SUBSET pairs each), flattened and concatenated.

Returns (train_dataset, val_dataset, test_dataset) as HuggingFace Datasets
with torch format set, ready for DataLoader.
"""

from __future__ import annotations

import logging
from typing import Tuple

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers import BertTokenizer

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_pairs(ds: Dataset, tokenizer: BertTokenizer) -> Dataset:
    """
    Convert a dataset of (chosen, rejected) pairs into individual labelled rows.

    Each input row has 'chosen' and 'rejected' string fields.
    Each output row has: input_ids, attention_mask, label (int), text (str).
    """
    def _process(batch):
        texts  = batch["chosen"] + batch["rejected"]
        labels = [1] * len(batch["chosen"]) + [0] * len(batch["rejected"])

        enc = tokenizer(
            texts,
            max_length=config.MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors=None,   # return plain lists; Dataset handles batching
        )
        return {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label":          labels,
            "text":           texts,
        }

    # Remove original string columns; keep only the flattened output.
    return ds.map(
        _process,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenising",
    )


def _sample_subset(subset_name: str, split: str, n: int, tokenizer: BertTokenizer) -> Dataset:
    """Load one HH-RLHF subset split, cap at n pairs, and flatten."""
    raw = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir=subset_name,
        split=split,
    )
    actual = len(raw)
    if actual < n:
        logger.warning(
            "%s/%s has only %d rows (requested %d); using all available.",
            subset_name, split, actual, n,
        )
        n = actual
    sampled = raw.shuffle(seed=config.SEED).select(range(n))
    return _flatten_pairs(sampled, tokenizer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_datasets(tokenizer: BertTokenizer | None = None) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Build train, val, and test datasets.

    Args:
        tokenizer: Optional pre-loaded BertTokenizer. If None, loads
                   bert-base-uncased automatically.

    Returns:
        (train_dataset, val_dataset, test_dataset) — all in torch format.
    """
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(config.MODEL_ID)

    logger.info("Loading HH-RLHF — helpful-base …")
    helpful_train = _sample_subset("helpful-base",  "train", config.HELPFUL_TRAIN_N,  tokenizer)
    helpful_test  = _sample_subset("helpful-base",  "test",  config.TEST_N_PER_SUBSET, tokenizer)

    logger.info("Loading HH-RLHF — harmless-base …")
    harmless_train = _sample_subset("harmless-base", "train", config.HARMLESS_TRAIN_N,  tokenizer)
    harmless_test  = _sample_subset("harmless-base", "test",  config.TEST_N_PER_SUBSET, tokenizer)

    # Combine and split training data 80/20.
    combined_train = concatenate_datasets([helpful_train, harmless_train])
    combined_train = combined_train.shuffle(seed=config.SEED)
    split = combined_train.train_test_split(test_size=config.VAL_SPLIT, seed=config.SEED)
    train_ds = split["train"]
    val_ds   = split["test"]

    # Test set is fully held-out (drawn from each subset's own test partition).
    test_ds = concatenate_datasets([helpful_test, harmless_test]).shuffle(seed=config.SEED)

    # Set torch format: input_ids and attention_mask as LongTensor, label as LongTensor.
    cols = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.with_format("torch", columns=cols, output_all_columns=True)
    val_ds   = val_ds.with_format("torch",   columns=cols, output_all_columns=True)
    test_ds  = test_ds.with_format("torch",  columns=cols, output_all_columns=True)

    logger.info(
        "Dataset sizes — train: %d  val: %d  test: %d",
        len(train_ds), len(val_ds), len(test_ds),
    )
    return train_ds, val_ds, test_ds
