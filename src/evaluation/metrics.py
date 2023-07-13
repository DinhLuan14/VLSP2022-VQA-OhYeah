# src/evaluation/metrics.py

import numpy as np
from datasets import load_metric


def compute_metrics(preds, labels):
    """Compute BLEU and ROUGE evaluation metrics.

    Args:
        preds (list): Generated predictions.
        labels (list): Ground truth labels.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """

    # Initialize scorers
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")

    # Compute scores
    bleu_score = bleu.compute(predictions=preds, references=labels)
    rouge_score = rouge.compute(predictions=preds, references=labels)

    # Extract scores
    bleu_score = bleu_score["bleu"]
    rouge_score = rouge_score["rougeL"].mid

    return {
        "bleu": bleu_score,
        "rouge": rouge_score,
    }


def compute_f1(preds, labels):
    """Compute F1 score for predictions vs labels.

    Args:
        preds (list): Generated predictions.
        labels (list): Ground truth labels.

    Returns:
        float: F1 score.
    """

    # Compute F1
    f1 = load_metric("f1")
    f1_score = f1.compute(predictions=preds, references=labels)["f1"]

    return f1_score
