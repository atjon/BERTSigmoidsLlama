# BERTSigmoidsLlama
CSE25 Project

A minimal mechanistic interpretability experiment. BERT (`bert-base-uncased`) is the analyzed model; Llama 3 8B (running locally via Ollama) is the explainer.

## Experiment

Subject: **dog** — 10 prompts (5 with `dog`, 5 matched controls) are run through BERT. Post-GELU neuron activations from layer 8's feed-forward network are captured via a PyTorch forward hook. The top differentially active neurons are identified and passed to a local Llama 3 8B model, which interprets whether those neurons plausibly encode the concept of `dog`.

## Setup

### 1. Install Ollama and pull Llama 3

Download Ollama from https://ollama.com/download, then:

```bash
ollama pull llama3
```

### 2. Install Python dependencies

For Apple Silicon (MPS-accelerated):

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (smaller download):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers ollama jupyter
```

### 3. Run the notebook

```bash
jupyter notebook experiment.ipynb
```

Run all cells top-to-bottom. The final cell prints a structured report with real activation values and the Llama 3 response as proof of execution.

## Stack

| Component | Role |
|---|---|
| `bert-base-uncased` | Analyzed model (BERT) |
| PyTorch forward hook on `layer[8].intermediate` | Activation capture |
| Apple MPS / CUDA / CPU | Inference acceleration |
| Llama 3 8B via Ollama | Explainer LLM |
