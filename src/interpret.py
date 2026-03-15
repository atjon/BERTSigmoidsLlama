"""
interpret.py — Neuron ranking, example attribution, and token attribution.

Two neuron ranking strategies:
    rank_neurons_by_variance()      — neurons with high activation variance across the
                                      corpus are likely more "discriminative" overall.
    rank_neurons_by_differential()  — neurons with the largest |chosen_mean - rejected_mean|
                                      directly capture the chosen/rejected distinction.
                                      (Ported directly from the proof-of-concept notebook.)

Token attribution:
    find_highly_activating_tokens() — for a single text, runs a fresh forward pass with
                                      an un-pooled hook, isolates per-token activation
                                      values for a given neuron, and returns the tokens
                                      whose activations exceed the 90th percentile.
                                      BERT subword tokens (## prefix) are merged back into
                                      full words before returning.

Example attribution:
    find_highly_activating_examples() — across the full corpus, returns samples where a
                                        given neuron's mean-pooled activation exceeds the
                                        90th percentile.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from src import config
from src.hooks import hooked_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neuron ranking
# ---------------------------------------------------------------------------

def rank_neurons_by_variance(activations: torch.Tensor) -> torch.Tensor:
    """
    Rank neurons by activation variance across N samples.

    Args:
        activations: (N, 3072) tensor.

    Returns:
        (3072,) tensor of neuron indices sorted descending by variance.
    """
    variance = activations.var(dim=0)       # (3072,)
    return variance.argsort(descending=True)


def rank_neurons_by_differential(
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> Dict:
    """
    Rank neurons by |chosen_mean - rejected_mean|.
    Ported directly from the proof-of-concept notebook.

    Args:
        activations: (N, 3072) tensor.
        labels:      (N,) tensor with 1=chosen, 0=rejected.

    Returns:
        Dict with keys:
            'indices'       — (TOP_N,) sorted neuron indices
            'diff'          — (3072,) chosen_mean - rejected_mean
            'chosen_mean'   — (3072,) per-neuron mean over chosen samples
            'rejected_mean' — (3072,) per-neuron mean over rejected samples
    """
    chosen_mask   = labels == 1
    rejected_mask = labels == 0

    chosen_mean   = activations[chosen_mask].mean(dim=0)    # (3072,)
    rejected_mean = activations[rejected_mask].mean(dim=0)  # (3072,)
    diff          = chosen_mean - rejected_mean              # (3072,)

    top_indices = diff.abs().argsort(descending=True)[: config.TOP_N_NEURONS]

    return {
        "indices":       top_indices,
        "diff":          diff,
        "chosen_mean":   chosen_mean,
        "rejected_mean": rejected_mean,
    }


# ---------------------------------------------------------------------------
# Example attribution
# ---------------------------------------------------------------------------

def find_highly_activating_examples(
    activations: torch.Tensor,
    labels: torch.Tensor,
    texts: List[str],
    neuron_idx: int,
    percentile: float = 90.0,
) -> List[Dict]:
    """
    Return samples where neuron_idx's mean-pooled activation > percentile threshold.

    Returns list of dicts: {'index': int, 'activation': float, 'label': int, 'text': str}
    sorted descending by activation.
    """
    neuron_acts = activations[:, neuron_idx]    # (N,)
    threshold   = torch.quantile(neuron_acts.float(), percentile / 100.0).item()

    results = []
    for i, (act, lbl, txt) in enumerate(zip(neuron_acts.tolist(), labels.tolist(), texts)):
        if act >= threshold:
            results.append({"index": i, "activation": act, "label": int(lbl), "text": txt})

    results.sort(key=lambda x: x["activation"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Token attribution (un-pooled forward pass)
# ---------------------------------------------------------------------------

def _merge_subword_tokens(
    token_strings: List[str],
    activation_values: List[float],
) -> Tuple[List[str], List[float]]:
    """
    Merge BERT ## subword tokens back into full words.

    Activation for a merged word = max of its constituent piece activations.
    Example: ['un', '##bel', '##iev', '##able'] → ['unbelievable']
    """
    merged_tokens: List[str]   = []
    merged_acts:   List[float] = []

    for tok, act in zip(token_strings, activation_values):
        if tok.startswith("##") and merged_tokens:
            merged_tokens[-1] = merged_tokens[-1] + tok[2:]
            merged_acts[-1]   = max(merged_acts[-1], act)
        else:
            merged_tokens.append(tok)
            merged_acts.append(act)

    return merged_tokens, merged_acts


class _TokenCapture:
    """Captures un-pooled activation tensor (1, seq_len, 3072) for a single forward pass."""
    def __init__(self):
        self.output: torch.Tensor | None = None

    def hook_fn(self, module, input, output):
        self.output = output.detach().cpu()   # (1, seq_len, 3072)


def find_highly_activating_tokens(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    layer_idx: int,
    neuron_idx: int,
    device: torch.device | None = None,
    percentile: float = 90.0,
) -> List[Tuple[str, float]]:
    """
    For a single text, find tokens whose activation for neuron_idx exceeds
    the percentile threshold within that sequence.

    Steps:
      1. Tokenise text (max 512 tokens, truncate right).
      2. Run forward pass with un-pooled hook on layer_idx.
      3. Extract per-token activation for neuron_idx.
      4. Filter tokens above the percentile threshold within the sequence.
      5. Merge BERT ## subword tokens.
      6. Skip special tokens ([CLS], [SEP], [PAD]).

    Returns:
        List of (token_string, activation_value) sorted descending by activation.
    """
    if device is None:
        device = config.get_device()

    tokenizer.truncation_side = "left"
    enc = tokenizer(
        text,
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)           # (1, seq_len)
    att_mask  = enc["attention_mask"].to(device)

    capture = _TokenCapture()
    target  = model.bert.encoder.layer[layer_idx].intermediate
    handle  = target.register_forward_hook(capture.hook_fn)

    model.eval()
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=att_mask)

    handle.remove()

    # capture.output: (1, seq_len, 3072)
    token_acts = capture.output[0, :, neuron_idx].tolist()   # (seq_len,)
    token_ids  = input_ids[0].tolist()
    tokens     = tokenizer.convert_ids_to_tokens(token_ids)

    # Remove special tokens before computing threshold.
    special = {"[CLS]", "[SEP]", "[PAD]"}
    filtered_tokens = []
    filtered_acts   = []
    for tok, act in zip(tokens, token_acts):
        if tok not in special:
            filtered_tokens.append(tok)
            filtered_acts.append(act)

    if not filtered_tokens:
        return []

    # Per-sequence percentile threshold.
    acts_t    = torch.tensor(filtered_acts)
    threshold = torch.quantile(acts_t, percentile / 100.0).item()

    high_tokens = [t for t, a in zip(filtered_tokens, filtered_acts) if a >= threshold]
    high_acts   = [a for a in filtered_acts if a >= threshold]

    # Merge subword pieces.
    merged_tokens, merged_acts = _merge_subword_tokens(high_tokens, high_acts)

    result = sorted(zip(merged_tokens, merged_acts), key=lambda x: x[1], reverse=True)
    return result


# ---------------------------------------------------------------------------
# Summary table builder (used by explain.py)
# ---------------------------------------------------------------------------

def build_neuron_table(
    diff_result: Dict,
    top_n: int | None = None,
) -> List[Dict]:
    """
    Build a list of dicts representing the neuron activation table for the
    explainer prompt.

    Each dict: {'rank', 'neuron', 'chosen_mean', 'rejected_mean', 'diff', 'direction'}
    """
    indices = diff_result["indices"]
    if top_n is not None:
        indices = indices[:top_n]

    rows = []
    for rank, idx in enumerate(indices.tolist(), start=1):
        d = diff_result["diff"][idx].item()
        rows.append({
            "rank":          rank,
            "neuron":        idx,
            "chosen_mean":   round(diff_result["chosen_mean"][idx].item(),   4),
            "rejected_mean": round(diff_result["rejected_mean"][idx].item(), 4),
            "diff":          round(d, 4),
            "direction":     "CHOSEN>" if d > 0 else "REJECT>",
        })
    return rows
