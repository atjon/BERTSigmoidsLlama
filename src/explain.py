"""
explain.py — Llama 3 8B (local via Ollama) explainer integration.

Builds a structured "summary" prompt containing:
  - The neuron activation table (chosen_mean, rejected_mean, diff, direction)
  - Up to 3 highly activating examples per top-5 neuron, with highlighted tokens
  - 4 analytical questions for the LLM

Two baselines:
  - Real prompt:     use the actual ranked neuron table.
  - Random baseline: shuffle the neuron table before sending to the LLM.
                     If the model gives equally confident-sounding explanations
                     for a shuffled table, the explanations are not meaningful.

Outputs are saved as JSON to outputs/explanations/.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import ollama

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_neuron_table(rows: List[Dict]) -> str:
    header = f"{'Rank':>4}  {'Neuron':>6}  {'Chosen':>9}  {'Rejected':>10}  {'Diff':>8}  Dir"
    sep    = "-" * len(header)
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f"{r['rank']:>4}  {r['neuron']:>6}  {r['chosen_mean']:>9.4f}"
            f"  {r['rejected_mean']:>10.4f}  {r['diff']:>8.4f}  {r['direction']}"
        )
    return "\n".join(lines)


def _format_examples(examples_by_neuron: Dict[int, List]) -> str:
    """
    examples_by_neuron: {neuron_idx: [(text, [(token, act), ...])}
    """
    parts = []
    for neuron_idx, examples in examples_by_neuron.items():
        parts.append(f"\n--- Neuron {neuron_idx} ---")
        for ex_idx, (text, high_tokens) in enumerate(examples, start=1):
            # Truncate long texts to 300 chars for readability in the prompt.
            excerpt = text[:300].replace("\n", " ") + ("…" if len(text) > 300 else "")
            token_str = ", ".join(
                f'"{t}" ({a:.3f})' for t, a in high_tokens[:10]
            )
            parts.append(f"  Example {ex_idx}: {excerpt}")
            if token_str:
                parts.append(f"  High-activation tokens (>90th pct): {token_str}")
    return "\n".join(parts)


def build_prompt(
    neuron_table_rows: List[Dict],
    examples_by_neuron: Dict[int, List],
    layer: int,
    is_baseline: bool = False,
) -> str:
    """
    Construct the full explainer prompt.

    Args:
        neuron_table_rows:  Output of interpret.build_neuron_table().
        examples_by_neuron: {neuron_idx: [(text, high_token_list), ...]}
        layer:              BERT layer that was hooked.
        is_baseline:        If True, shuffle neuron_table_rows before formatting.
    """
    rows = list(neuron_table_rows)   # copy so we don't mutate caller's data
    if is_baseline:
        random.shuffle(rows)
        # Re-assign ranks so the table still looks coherent to the LLM.
        for i, r in enumerate(rows, start=1):
            r = dict(r)
            r["rank"] = i
            rows[i - 1] = r

    table_str   = _format_neuron_table(rows)
    example_str = _format_examples(examples_by_neuron)

    prompt = f"""You are a mechanistic interpretability researcher analysing a BERT model \
(bert-base-uncased) that has been fine-tuned as a binary preference classifier on \
human feedback data.  The classifier predicts whether a response to a human prompt \
is "chosen" (preferred, label 1) or "rejected" (not preferred, label 0).

The activations below come from layer {layer}'s feed-forward network (BertIntermediate: \
linear 768→3072 + GELU).  Each row is one neuron; activations are mean-pooled over \
sequence tokens.

NEURON ACTIVATION TABLE  (top {len(rows)} neurons ranked by |chosen_mean - rejected_mean|)
{table_str}

HIGHLY ACTIVATING EXAMPLES  (samples where the neuron's activation > 90th percentile)
Text excerpts are shown with their highest-activating tokens (>90th percentile within \
that sequence).
{example_str}

QUESTIONS — please answer each in 2–4 sentences:

1. What semantic or pragmatic features do these neurons appear to encode?  \
Are they capturing surface form, syntactic structure, sentiment, formality, \
or something else?

2. What distinguishes "chosen" from "rejected" responses as seen through these neurons?  \
What qualities do the CHOSEN> neurons reward?

3. Do the highly activating tokens suggest the neurons respond to specific words/phrases, \
or to broader contextual patterns?  Give concrete examples from the table.

4. What alternative hypotheses could explain these activation patterns, and what follow-up \
experiment would help disambiguate them?
"""
    return prompt


# ---------------------------------------------------------------------------
# Ollama interaction
# ---------------------------------------------------------------------------

def _query_llama(prompt: str) -> str:
    """Send prompt to local Llama 3 8B via Ollama; return response text."""
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def explain_neurons(
    neuron_table_rows: List[Dict],
    examples_by_neuron: Dict[int, List],
    layer: int = config.INTERP_LAYER,
) -> Dict:
    """
    Run both the real and random-baseline explanations, save to JSON.

    Returns:
        Dict with 'prompt', 'response', 'baseline_prompt', 'baseline_response',
        'timestamp', 'layer', 'output_path'.
    """
    logger.info("Querying Llama for real explanation …")
    real_prompt    = build_prompt(neuron_table_rows, examples_by_neuron, layer, is_baseline=False)
    real_response  = _query_llama(real_prompt)

    logger.info("Querying Llama for random-baseline explanation …")
    base_prompt    = build_prompt(neuron_table_rows, examples_by_neuron, layer, is_baseline=True)
    base_response  = _query_llama(base_prompt)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp":        timestamp,
        "layer":            layer,
        "prompt":           real_prompt,
        "response":         real_response,
        "baseline_prompt":  base_prompt,
        "baseline_response": base_response,
    }

    out_path = os.path.join(config.EXPLANATIONS_DIR, f"explanation_{timestamp}.json")
    os.makedirs(config.EXPLANATIONS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    result["output_path"] = out_path
    logger.info("Explanation saved to %s", out_path)
    return result
