"""
config.py — Single source of truth for all hyperparameters and path constants.
Edit values here; all other modules import from this file.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "bert-base-uncased"

# ---------------------------------------------------------------------------
# Dataset sampling
# ---------------------------------------------------------------------------
HELPFUL_TRAIN_N    = 36_000   # pairs sampled from helpful-base train split
HARMLESS_TRAIN_N   = 36_000   # pairs sampled from harmless-base train split
TEST_N_PER_SUBSET  = 4_000    # pairs sampled from each subset's test split
VAL_SPLIT          = 0.2      # fraction of combined train data held out for val
SEED               = 42

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 512   # BERT hard limit; the "~500 token" spec rounds to this

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_EPOCHS   = 3
BATCH_SIZE     = 16
LR             = 2e-5
WARMUP_RATIO   = 0.06    # fraction of total steps used for linear warmup
GRAD_CLIP      = 1.0

# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------
INTERP_LAYER    = 8      # BertIntermediate layer to hook (0-indexed)
                          # Justified by Tenney et al. (2019): semantic/entity
                          # representations peak at layers 7–9 in BERT.
TOP_N_NEURONS   = 30     # neurons to include in the explainer prompt
PERCENTILE_HIGH = 90.0   # threshold for "highly activating" (token/example level)
INTERP_SAMPLE_N = 500    # test samples to run through the interpretability pipeline
                          # Stratified: 250 chosen + 250 rejected.

# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR  = os.path.join(_ROOT, "checkpoints")
ACTIVATIONS_DIR = os.path.join(_ROOT, "outputs", "activations")
EXPLANATIONS_DIR = os.path.join(_ROOT, "outputs", "explanations")

for _d in (CHECKPOINT_DIR, ACTIVATIONS_DIR, EXPLANATIONS_DIR):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    """Return the best available device: MPS → CUDA → CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
