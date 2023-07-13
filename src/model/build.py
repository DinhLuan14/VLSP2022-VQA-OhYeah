# src/model/build.py

import torch
from transformers import AutoModelForSeq2SeqLM


def build_model(config):
    """Build the Encoder-Decoder model along with tokenizer.

    Args:
        config (dict): Dictionary containing model configurations.

    Returns:
        model (EncoderDecoderModel): Encoder-Decoder model
        tokenizer (AutoTokenizer): Tokenizer for encoding text.
    """

    # Load model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Send model to device
    model = model.to(config["device"])

    return model, tokenizer
