"""
train.py — Fine-tuning loop, evaluation, and checkpointing.

train()    — fine-tunes BertForSequenceClassification for TRAIN_EPOCHS epochs,
             saving a checkpoint after each epoch and returning a training
             history dict for plotting.

evaluate() — computes accuracy, F1, and a full sklearn classification report
             on any labelled dataset.  Also computes the majority-class baseline
             (always predict chosen = label 1) for direct comparison.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from datasets import Dataset
from sklearn.metrics import classification_report, accuracy_score

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
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
    Fine-tune model on train_ds with validation after each epoch.

    Args:
        model:     BertForSequenceClassification in train mode on device.
        train_ds:  HuggingFace Dataset with torch format (input_ids, attention_mask, label).
        val_ds:    Validation dataset in the same format.
        tokenizer: Saved alongside each checkpoint so load_checkpoint works standalone.
        device:    Target device. Defaults to config.get_device().
        start_epoch: Epoch index to start/resume from.

    Returns:
        History dict: {'epoch_numbers': [...], 'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}
        (one entry per epoch that was run in this call)
    """
    if device is None:
        device = config.get_device()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # 0 avoids issues with MPS + multiprocessing fork
    )
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE * 2, num_workers=0)

    optimizer = AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)

    total_steps   = len(train_loader) * max(config.TRAIN_EPOCHS - start_epoch, 1)
    warmup_steps  = int(total_steps * config.WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

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
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
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
        val_metrics = evaluate(model, val_ds, device, verbose=False)
        logger.info(
            "Epoch %d — train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, avg_train_loss, val_metrics["loss"], val_metrics["accuracy"],
        )

        history["epoch_numbers"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        # ---- checkpoint ----
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"epoch-{epoch}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        logger.info("Checkpoint saved to %s", ckpt_path)

    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: BertForSequenceClassification,
    dataset: Dataset,
    device: torch.device | None = None,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model on dataset.

    Returns a dict with:
      - accuracy: float
      - loss: float
      - report: sklearn classification_report string
      - majority_class_accuracy: float  (always-predict-chosen baseline)
      - predictions: List[int]
      - labels: List[int]
    """
    if device is None:
        device = config.get_device()

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, num_workers=0)
    model.eval()

    all_preds:  List[int]   = []
    all_labels: List[int]   = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    report   = classification_report(
        all_labels, all_preds,
        target_names=["rejected", "chosen"],
        digits=4,
    )

    # Majority-class baseline: always predict chosen (label=1).
    # Dataset is balanced 50/50 by construction, so this ≈ 50%.
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
