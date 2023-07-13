# src/train.py

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from src.model import build_model
from src.dataset import get_datasets
from src.evaluation import compute_metrics
from src.utils import set_seed
from dataset.evaluate_metrics_new import my_compute_f1, my_compute_bleu
from transformers import EarlyStoppingCallback
from transformers import ViTFeatureExtractor, ViTModel
from polyglot.detect import Detector
from sklearn.model_selection import GroupKFold
from datasets import Image, Dataset
import json
from polyglot.detect.base import logger as polyglot_logger
from pathlib import Path
import warnings
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from datasets import Image, Dataset
import PIL
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from datetime import datetime
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
import random
import copy
import re
import sys
import gc
import os
from src.utils import set_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152


warnings.simplefilter(action="ignore", category=FutureWarning)


ROOT_PATH = os.getcwd()
DATA_PATH = f"{ROOT_PATH}/data"
# DATASET_PATH = f"{ROOT_PATH}/dataset"
MODEL_PATH = f"{ROOT_PATH}/model"
GRAPH_DATA_PATH = f"{ROOT_PATH}/GraphData"
OUTPUT_PATH = f"{ROOT_PATH}/output"
DATASET_PATH = "/data4/share_nlp/data/luannd/vlsp/dataset"
print(DATASET_PATH)


class DataCollator:
    def __init__(self, img_model, feature_extractor, tokenizer, text_embed_model):
        self.img_model = img_model
        self.tokenizer = tokenizer
        self.text_embed_model = text_embed_model
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        # tokenize the inputs and labels
        image = [i["image"] for i in batch]
        question = [i["lang"] + ": " + i["question"] for i in batch]
        # question = [i['question'] for i in batch]
        answer = [i["answer"] for i in batch]
        ques_inputs = self.tokenizer(
            question,
            max_length=question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            ques_embeds = (
                self.text_embed_model(ques_inputs["input_ids"].to(device))
                .cpu()
                .detach()
            )
            image_inputs = self.feature_extractor(image, return_tensors="pt")
            for u, v in image_inputs.items():
                image_inputs[u] = v.to(device)
            img_embeds = self.img_model(**image_inputs).last_hidden_state.cpu().detach()
            inputs_embeds = torch.cat((ques_embeds, img_embeds), 1)

        clear_gpu()
        attention_mask = torch.cat(
            (
                ques_inputs.attention_mask,
                torch.ones(img_embeds.shape[0], img_embeds.shape[1]),
            ),
            1,
        )
        del image_inputs, ques_embeds, img_embeds
        outputs = self.tokenizer(
            answer,
            max_length=answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = outputs.input_ids.clone()
        #         labels = torch.where(labels== tokenizer.pad_token_id, -100, labels)
        labels[labels == tokenizer.pad_token_id] = -100
        # labels = labels.roll(-1, 1)
        # labels[:, -1] = -100

        result = {}
        result["labels"] = labels
        result["inputs_embeds"] = inputs_embeds
        result["attention_mask"] = attention_mask
        #         result["decoder_input_ids"] = outputs.input_ids
        #         result["decoder_attention_mask"] = outputs.attention_mask
        return result


class TestCollator:
    def __init__(self, img_model, feature_extractor, tokenizer, text_embed_model):
        self.img_model = img_model
        self.tokenizer = tokenizer
        self.text_embed_model = text_embed_model
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        # tokenize the inputs and labels
        image = [i["image"] for i in batch]
        question = [i["lang"] + ": " + i["question"] for i in batch]
        #         question = [i['question'] for i in batch]
        answer = [i["answer"] for i in batch]
        ques_inputs = self.tokenizer(
            question,
            max_length=question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            ques_embeds = (
                self.text_embed_model(ques_inputs["input_ids"].to(device))
                .cpu()
                .detach()
            )

            image_inputs = self.feature_extractor(image, return_tensors="pt")
            for u, v in image_inputs.items():
                image_inputs[u] = v.to(device)
            img_embeds = self.img_model(**image_inputs).last_hidden_state.cpu().detach()

            inputs_embeds = torch.cat((ques_embeds, img_embeds), 1)

        clear_gpu()
        attention_mask = torch.cat(
            (
                ques_inputs.attention_mask,
                torch.ones(img_embeds.shape[0], img_embeds.shape[1]),
            ),
            1,
        )
        del image_inputs, ques_embeds, img_embeds

        result = {}
        result["inputs_embeds"] = inputs_embeds
        result["attention_mask"] = attention_mask

        return result


def train():
    """Main training routine."""

    # 1. Setup
    config = {"seed": 42, "model_name": "baseline"}
    set_seed(config["seed"])

    # 2. Load data
    train_data, eval_data = get_datasets(config)

    # 3. Load model and tokenizer
    model, tokenizer = build_model(config)

    clear_gpu()

    training_steps = int(len(train_ds) / (CFG.batch_size * CFG.grad_accm_steps))
    eval_steps = int(training_steps / CFG.eval_in_epoch)
    print(f"eval_steps: {eval_steps} | training_steps {training_steps}")

    print("training:....")
    training_args = Seq2SeqTrainingArguments(
        output_dir="final_vqa_v3_full_data",
        evaluation_strategy="steps",
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        predict_with_generate=True,
        num_train_epochs=CFG.num_train_epochs,
        # optim = 'adamw_torch',
        gradient_accumulation_steps=2,
        logging_steps=200,
        save_steps=200,
        eval_steps=200,
        # warmup_steps=0,
        overwrite_output_dir=True,
        # fp16=True,
        report_to="wandb",
        remove_unused_columns=False,
        save_total_limit=2,
        # push_to_hub=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
    )

    training_args.weight_decay = CFG.weight_decay
    # training_args.learning_rate = CFG.learning_rate
    training_args.max_grad_norm = CFG.max_grad_norm
    training_args.warmup_ratio = CFG.warmup_ratio

    train_collator = DataCollator(vit_model, feature_extractor, tokenizer, embed_model)
    test_collator = TestCollator(vit_model, feature_extractor, tokenizer, embed_model)
    compute_metrics = compute_metrics

    # 6. Initialize Trainer
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds.select(range(50)),
        data_collator=train_collator,
    )

    # 7. Train model
    trainer.train()


if __name__ == "__main__":
    train()
