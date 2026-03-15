"""
train.py — Fine-tuning loop, evaluation, and checkpointing.

train()              — fine-tunes the model with a pairwise ranking loss:
                       loss = -log(sigmoid(score_chosen - score_rejected))
                       Expects pair-format batches from load_pair_datasets().

evaluate_pairwise()  — computes pairwise accuracy on a pair-format dataset
                       (fraction of pairs where score_chosen > score_rejected).
                       Used for validation during training.

evaluate()           — computes per-row accuracy and classification report on a
                       flat (individual-row) dataset. Used for the interpretability
                       pipeline's test-set evaluation.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training (pairwise ranking loss)
# ---------------------------------------------------------------------------

def train(
    model: BertForSequenceClassification,
    train_ds: Dataset,
    val_ds: Dataset,
    tokenizer: BertTokenizer,
    device: torch.device | None = None,
    start_epoch: int = 0,
) -> Dict[str, List[float]]:
    """
    Fine-tune model on train_ds using pairwise ranking loss.

    Args:
        model:       BertForSequenceClassification (num_labels=1) in train mode on device.
        train_ds:    Pair-format Dataset with fields:
                     chosen_input_ids, chosen_attention_mask,
                     rejected_input_ids, rejected_attention_mask.
        val_ds:      Same pair format; evaluated with evaluate_pairwise() after each epoch.
        tokenizer:   Saved alongside each checkpoint.
        device:      Target device. Defaults to config.get_device().
        start_epoch: Epoch index to start/resume from.

    Returns:
        History dict: {'epoch_numbers', 'train_loss', 'val_loss', 'val_acc'}
    """
    if device is None:
        device = config.get_device()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)

    total_steps  = len(train_loader) * max(config.TRAIN_EPOCHS - start_epoch, 1)
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history: Dict[str, List[float]] = {
        "epoch_numbers": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    if start_epoch >= config.TRAIN_EPOCHS:
        logger.info("Training already complete at epoch %d; nothing to run.", start_epoch)
        return history

    for epoch in range(start_epoch, config.TRAIN_EPOCHS):
        # ---- train ----
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            chosen_ids    = batch["chosen_input_ids"].to(device)
            chosen_mask   = batch["chosen_attention_mask"].to(device)
            rejected_ids  = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            optimizer.zero_grad()

            chosen_scores   = model(input_ids=chosen_ids,   attention_mask=chosen_mask).logits.squeeze(-1)
            rejected_scores = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)

            # Bradley-Terry ranking loss: -log(sigmoid(score_chosen - score_rejected))
            loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if (step + 1) % 200 == 0:
                avg = running_loss / (step + 1)
                logger.info("Epoch %d  step %d/%d  loss=%.4f", epoch, step + 1, len(train_loader), avg)

        avg_train_loss = running_loss / len(train_loader)

        # ---- validate ----
        val_metrics = evaluate_pairwise(model, val_ds, device, verbose=False)
        logger.info(
            "Epoch %d — train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, avg_train_loss, val_metrics["loss"], val_metrics["pairwise_accuracy"],
        )

        history["epoch_numbers"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["pairwise_accuracy"])

        # ---- checkpoint ----
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch-{epoch}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        logger.info("Checkpoint saved to %s", ckpt_path)

    return history


# ---------------------------------------------------------------------------
# Pairwise evaluation (used during training on val_pair_ds)
# ---------------------------------------------------------------------------

def evaluate_pairwise(
    model: BertForSequenceClassification,
    dataset: Dataset,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model on a pair-format dataset.

    Returns a dict with:
      - pairwise_accuracy: fraction of pairs where score_chosen > score_rejected
      - loss: mean ranking loss
    """
    if device is None:
        device = config.get_device()

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, num_workers=0)
    model.eval()

    total_loss   = 0.0
    total_correct = 0
    total_pairs   = 0

    with torch.no_grad():
        for batch in loader:
            chosen_ids    = batch["chosen_input_ids"].to(device)
            chosen_mask   = batch["chosen_attention_mask"].to(device)
            rejected_ids  = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            chosen_scores   = model(input_ids=chosen_ids,   attention_mask=chosen_mask).logits.squeeze(-1)
            rejected_scores = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)

            loss = -F.logsigmoid(chosen_scores - rejected_scores).mean()
            total_loss += loss.item()

            total_correct += (chosen_scores > rejected_scores).sum().item()
            total_pairs   += chosen_scores.size(0)

    avg_loss = total_loss / len(loader)
    pair_acc = total_correct / total_pairs

    if verbose:
        print(f"\nPairwise accuracy : {pair_acc:.4f}")
        print(f"Ranking loss      : {avg_loss:.4f}")

    return {"pairwise_accuracy": pair_acc, "loss": avg_loss}


# ---------------------------------------------------------------------------
# Flat evaluation (used for test set in the interpretability pipeline)
# ---------------------------------------------------------------------------

def evaluate(
    model: BertForSequenceClassification,
    dataset: Dataset,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model on a flat (individual-row, binary-label) dataset.

    The model outputs a scalar score (num_labels=1); we treat score > 0 as
    predicting "chosen" (label=1) and score <= 0 as "rejected" (label=0).

    Returns a dict with:
      - accuracy, loss, report, majority_class_accuracy, predictions, labels
    """
    if device is None:
        device = config.get_device()

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, num_workers=0)
    model.eval()

    all_preds:  List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            scores = model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

            # Ranking loss against a zero reference (no explicit rejected pair here)
            # Use BCE against label as a proxy loss for logging only.
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
            total_loss += loss.item()

            preds = (scores > 0).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss     = total_loss / len(loader)
    acc          = accuracy_score(all_labels, all_preds)
    report       = classification_report(
        all_labels, all_preds,
        target_names=["rejected", "chosen"],
        digits=4,
    )
    majority_acc = sum(1 for l in all_labels if l == 1) / len(all_labels)

    if verbose:
        print(f"\nAccuracy       : {acc:.4f}")
        print(f"Loss           : {avg_loss:.4f}")
        print(f"Majority-class : {majority_acc:.4f} (always-predict-chosen)")
        print("\nClassification report:")
        print(report)

    return {
        "accuracy":                acc,
        "loss":                    avg_loss,
        "report":                  report,
        "majority_class_accuracy": majority_acc,
        "predictions":             all_preds,
        "labels":                  all_labels,
    }
