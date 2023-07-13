# src/model/predict.py

import torch
from transformers import AutoTokenizer


def get_predictions(model, dataloader, tokenizer, config):
    """Generate predictions for a dataloader using the model.

    Args:
        model (EncoderDecoderModel): Encoder-Decoder model.
        dataloader (Dataloader): PyTorch dataloader.
        tokenizer (AutoTokenizer): Tokenizer to decode predictions.
        config (dict): Dict containing inference configs like max length.

    Returns:
        preds (list): List of generated predictions.
    """

    # Put model in eval mode
    model.eval()

    preds = []

    # Iterate over dataloader
    for batch in dataloader:
        # Move batch to device
        inputs = {k: v.to(config["device"]) for k, v in batch.items()}

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=config["max_length"])

        # Decode and store predictions
        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(batch_preds)

    return preds
