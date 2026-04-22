# sentiment_engine.py — FinBERT NLP scorer
# ─────────────────────────────────────────────────────────────
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import Literal
from loguru import logger
import torch
import config

@dataclass
class SentimentResult:
    label:      Literal[
'bullish', 'bearish', 'neutral']
    score:      float   # positive = +confidence, negative = -confidence
    confidence: float   # raw model confidence 0.0 – 1.0
    text:       str

# ── Load FinBERT once at module import ────────────────────────
# First run: downloads ~420MB. Subsequent runs: loads from cache.
# Model is hosted by ProsusAI on HuggingFace — trained on
# financial news text, far more accurate than VADER for headlines.
logger.info(f"Loading FinBERT on {config.DEVICE}...")

_MODEL_NAME = 'ProsusAI/finbert'
_tokenizer  = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model      = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
_model.to(config.DEVICE)  # move model to GPU (or CPU if unavailable)
_model.eval()             # set to evaluation mode (disables dropout)

# FinBERT label → our label mapping
_LABEL_MAP = {'positive': 'bullish', 'negative': 'bearish', 'neutral': 'neutral'}

logger.success(f"FinBERT loaded on {config.DEVICE} ✓")


def score_text(text: str) -> SentimentResult:
    """Score a single headline with FinBERT."""
    if not text or not text.strip():
        return SentimentResult(
'neutral', 0.0, 0.0, text)

    # Tokenise — max 512 tokens (FinBERT hard limit)
    inputs = _tokenizer(
        text[:512],
        return_tensors  = 'pt',
        truncation      = True,
        max_length      = 512,
        padding         = True
    ).to(config.DEVICE)

    with torch.no_grad():           # disable gradient computation for speed
        outputs = _model(**inputs)
        probs   = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # FinBERT outputs: [negative, neutral, positive]
    labels     = _model.config.id2label
    pred_idx   = probs[0].argmax().item()
    confidence = probs[0][pred_idx].item()
    raw_label  = labels[pred_idx]  # 'positive', 'negative', or 'neutral'
    our_label  = _LABEL_MAP[raw_label]

    # Filter low-confidence predictions — treat as neutral
    if confidence < config.MIN_CONFIDENCE:
        our_label = 'neutral'

    score = confidence if our_label == 'bullish' \
       else -confidence if our_label == 'bearish' \
       else 0.0

    return SentimentResult(our_label, score, confidence, text)


def score_batch(texts: list[str]) -> list[SentimentResult]:
    """Score a list of texts. Filters empty strings automatically."""
    return [score_text(t) for t in texts if t and t.strip()]