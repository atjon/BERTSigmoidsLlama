# BERTSigmoidsLlama

Mechanistic interpretability of a BERT preference classifier trained on Anthropic HH-RLHF data.
`bert-base-uncased` is the subject model; Llama 3 8B (locally via Ollama) is the explainer.

## Project structure

```
experiment.ipynb      Proof-of-concept (10 toy prompts, vanilla BERT) — preserved, untouched
analysis.ipynb        Full project pipeline — run this for the real results

src/
  config.py           All hyperparameters and paths (edit here)
  data.py             HH-RLHF loading, tokenisation, train/val/test splits
  model.py            BertForSequenceClassification + checkpoint helpers
  train.py            Training loop, evaluation, majority-class baseline
  hooks.py            Forward hook registration, activation capture & caching
  interpret.py        Neuron ranking, token attribution, subword merging
  explain.py          Llama 3 8B prompt builder + Ollama integration

checkpoints/          Fine-tuned weights per epoch (gitignored)
outputs/
  activations/        Cached activation tensors (gitignored)
  explanations/       LLM explanation JSON files (committed)
```

## Pipeline

1. **Data** — 36k helpful-base + 36k harmless-base preference pairs → 144k labelled rows (50/50 chosen/rejected) → 80/20 train/val split → 8k held-out test pairs.
2. **Fine-tune** — `bert-base-uncased` binary classifier (linear head on CLS token), 3 epochs, AdamW + linear warmup.
3. **Evaluate** — accuracy / F1 vs. majority-class baseline (~50%).
4. **Capture** — forward hooks on layer-8 BertIntermediate capture 500 stratified test activations.
5. **Rank** — neurons ranked by `|chosen_mean − rejected_mean|` and by activation variance.
6. **Attribute** — per-token activations for top neurons; BERT subword tokens merged.
7. **Explain** — Llama 3 8B reads the neuron table + highlighted examples and answers 4 interpretability questions. Random-baseline variant (shuffled table) tests whether explanations are data-dependent.

## Setup

### 1. Install Ollama and pull Llama 3

```bash
# Download from https://ollama.com/download, then:
ollama pull llama3
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
jupyter notebook analysis.ipynb
```

Run all cells top-to-bottom. If `checkpoints/epoch-2/` already exists, training is skipped and the checkpoint is loaded automatically.

### 4. Proof-of-concept (optional)

The original 10-prompt toy experiment is preserved in `experiment.ipynb`. It requires no dataset and completes in seconds.

## Stack

| Component | Role |
|---|---|
| `bert-base-uncased` | Subject model (fine-tuned as classifier) |
| Anthropic HH-RLHF | Training data (helpful + harmless preference pairs) |
| `BertForSequenceClassification` | Binary classification head (CLS token → 2-class linear) |
| PyTorch forward hook on `layer[8].intermediate` | Activation capture (768→3072 FFN, post-GELU) |
| Apple MPS / CUDA / CPU | Inference and training acceleration |
| Llama 3 8B via Ollama | Explainer LLM |
| scikit-learn | Evaluation metrics |
| HuggingFace `datasets` | Dataset loading and caching |
