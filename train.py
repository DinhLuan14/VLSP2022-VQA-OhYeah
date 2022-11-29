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
print("imported 1/2")
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold
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
os.environ['HTTPS_PROXY'] = 'http://10.60.28.99:81'
os.environ['HTTP_PROXY'] = 'http://10.60.28.99:81'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


warnings.simplefilter(action='ignore', category=FutureWarning)


ROOT_PATH = os.getcwd()
DATA_PATH = f"{ROOT_PATH}/data"
# DATASET_PATH = f"{ROOT_PATH}/dataset"
MODEL_PATH = f"{ROOT_PATH}/model"
GRAPH_DATA_PATH = f"{ROOT_PATH}/GraphData"
OUTPUT_PATH = f"{ROOT_PATH}/output"
DATASET_PATH = '/data4/share_nlp/data/luannd/vlsp/dataset'
print(DATASET_PATH)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(37)


class Config:
    def __init__(self):
        self.kfold = 10
        self.fold = 0
        self.epochs = 300
        self.batch_size = 16
        self.seed = 37

        self.num_train_epochs = 10
        self.eval_in_epoch = 10

        self.learning_rate = 3e-4
        self.dropout = 0.1
        self.weight_decay = 0.01
        self.max_grad_norm = 3.0
        self.warmup_ratio = 0.2
        self.grad_accm_steps = 1
        self.freeze_embeddings = False

        self.vit_model = "/data4/share_nlp/data/luannd/pretrained_model/vit-base-patch16-224-in21k"
        self.lm_model = "/data4/share_nlp/data/luannd/pretrained_model/mt5-base"

        self.train_img_dir = f'{DATASET_PATH}/train-images/'
        self.train_json_path = f'{DATASET_PATH}/evjvqa_train.json'

        self.test_img_dir = f'{DATASET_PATH}/public-test-images/'
        self.test_json_path = f'{DATASET_PATH}/evjvqa_public_test.json'
        self.IN_DIR = IN_DIR = Path(DATASET_PATH)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')


CFG = Config()
device = CFG.device
device


# os.environ["WANDB_DISABLED"] = "true"
warnings.simplefilter(action='ignore', category=FutureWarning)
polyglot_logger.setLevel("ERROR")


def show_image(image_file):
    image = PIL.Image.open(image_file)
    return image


def detect_lang(s):
    try:
        return str(Detector(s).language.code)
    except Exception as e:
        print(f"error: {e}\n{s}")
        return "unk"


def is_japanese_sentence(text: str):
    pattern = r"[\u3000-\u303F]|[\u3040-\u309F]|[\u30A0-\u30FF]|[\uFF00-\uFFEF]|[\u4E00-\u9FAF]|[\u2605-\u2606]|[\u2190-\u2195]|\u203B"
    return re.search(pattern, text) is not None


def replace_word(lang, length, text):
    if (lang == "ja") & (length == 1):
        dicts = {
            "2": "二",
            "6": "六",
            "7": "七"
        }
        for key, value in dicts.items():
            text = text.replace(value, key)
        return text
    elif (lang == "vi") & (length == 1):
        dicts = {
            "hai": "2",
            "ba": "3",
        }
        for key, value in dicts.items():
            text = text.replace(value, key)
        return text
    else:
        return text


def read_df(image_folder, data_file, train=False):
    black_id = [1493, 2397, 2900, 2913, 2952, 2955, 2956, 2959, 2989, 4094, 10008,
                10009,
                10012,
                10082,
                10083,
                10013,
                10014,
                10015,
                10088,
                10018,
                10089,
                10091,
                10022,
                10092,
                10093,
                10023,
                10024,
                10097,
                10028,
                10098,
                10099,
                10101,
                10030,
                10031,
                10033,
                10104,
                10034,
                10035,
                10108,
                10109,
                10110,
                10111,
                10112,
                10113,
                10114,
                10115,
                10044,
                10048,
                10117,
                10118,
                10119,
                10050,
                10051,
                10121,
                10052,
                10125,
                10054,
                10056,
                10127,
                10129,
                10130,
                10131,
                10132,
                10133,
                10134,
                10064,
                10065,
                10135,
                10138,
                10068,
                10069,
                10139,
                10071,
                10142,
                10144,
                10145,
                10146,
                10075,
                10148,
                10149,
                10150,
                10151,
                18584,
                20381,
                21394, 1119, 16850, 22091, 12154, 23370, 23516]
    f = open(data_file)
    data = json.load(f)
    print("length data after preprocessing: ", len(data["annotations"]))

    img_df = pd.DataFrame.from_records(data['images'])
    ann_df = pd.DataFrame.from_records(data['annotations'])
    img_df.rename(columns={"id": "image_id"}, inplace=True)
    img_df['image'] = img_df['filename'].apply(
        lambda x: f"{image_folder}/{x}"
    )
    df = pd.merge(img_df, ann_df, on="image_id")
    df["lang"] = df["question"].apply(lambda x: detect_lang(x))
    df["length_ans"] = df["answer"].apply(lambda x: len(
        list(x)) if is_japanese_sentence(x) else len(x.split()))
    if train:
        # q&a not same language
        df = df[~df['id'].isin(black_id)]

        df["answer"] = df.apply(lambda x: replace_word(
            x.lang, x.length_ans, x.answer), axis=1)
        print("length data before preprocessing: ", len(df))
        return df.reset_index()
    else:
        return df.reset_index()


def df_to_dataset(df):
    dataset = Dataset.from_pandas(
        df, preserve_index=False).cast_column("image", Image())
    dataset = dataset.remove_columns(['filename', 'length_ans'])
    return dataset


df = read_df(CFG.IN_DIR/'train-images', CFG.IN_DIR/'evjvqa_train.json', True)

gkf = GroupKFold(n_splits=CFG.kfold)
for fold, (_, val_) in enumerate(gkf.split(X=df, groups=df.image_id)):
    df.loc[val_, "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)
train_df = df[df["kfold"] != CFG.fold]
eval_df = df[df["kfold"] == CFG.fold]

train_ds = df_to_dataset(train_df).remove_columns(['kfold'])
eval_ds = df_to_dataset(eval_df).remove_columns(['kfold'])
print('train_ds', len(train_ds))
print('eval_ds', len(eval_ds))

test_df = read_df(CFG.IN_DIR/'public-test-images',
                  CFG.IN_DIR/'evjvqa_public_test.json')
test_ds = df_to_dataset(test_df)
print('test_ds', len(test_ds))
private_df = read_df(CFG.IN_DIR/'private-test-images',
                     CFG.IN_DIR/'prepared_evjvqa_private_test.json')
private_ds = df_to_dataset(private_df)
print('private_ds', len(private_ds))


print('load text model: ')
model = MT5ForConditionalGeneration.from_pretrained(CFG.lm_model)
tokenizer = T5Tokenizer.from_pretrained(CFG.lm_model)
model.encoder.main_input_name = 'inputs_embeds'
embed_model = copy.deepcopy(model.shared).to(device)

print('load vision model: ')
feature_extractor = ViTFeatureExtractor.from_pretrained(
    '/data4/share_nlp/data/luannd/pretrained_model/vit-base-patch16-224-in21k')

vit_model = ViTModel.from_pretrained(
    '/data4/share_nlp/data/luannd/pretrained_model/vit-base-patch16-224-in21k')
vit_model = vit_model.to(device)


question_length = 60
answer_length = 40


def mean_seq_length(batch):
    lens = [np.sum(np.array(seq) != tokenizer.pad_token_id) for seq in batch]
    return np.mean(lens)


def postprocess(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Convert Padding token in label
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Calculate mean length of preds and labels
    mean_label_len = mean_seq_length(labels)
    mean_pred_len = mean_seq_length(preds)

    # Decoded Preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]

    # Decoded Labels
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_labels = [label.strip() for label in decoded_labels]

    return decoded_preds, decoded_labels, mean_pred_len, mean_label_len


def compute_metrics(eval_preds):
    # Decoded preds to words & get mean length
    decoded_preds, decoded_labels, mean_pred_len, mean_label_len = postprocess(
        eval_preds)

    # Calculate BLEU & F1
    bleu_score = my_compute_bleu(decoded_labels, decoded_preds)
    f1_score = my_compute_f1(decoded_labels, decoded_preds)

    result = {
        'f1_score': f1_score,
        'bleu_score': bleu_score,
        # 'precision': precision,
        # 'recall': recall,
        'mean_pred_len': mean_pred_len,
        'mean_label_len': mean_label_len
    }

    result = {k: round(v, 4) for k, v in result.items()}

    for idx in np.random.randint(0, len(decoded_preds), size=10):
        print('-'*35)
        print(f"{idx} - Label: {decoded_labels[idx]}")
        print(f"{idx} - Predict: {decoded_preds[idx]}")

    return result


class DataCollator:
    def __init__(self, img_model, feature_extractor, tokenizer, text_embed_model):
        self.img_model = img_model
        self.tokenizer = tokenizer
        self.text_embed_model = text_embed_model
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        # tokenize the inputs and labels
        image = [i['image'] for i in batch]
        question = [i['lang'] + ': ' + i['question'] for i in batch]
        # question = [i['question'] for i in batch]
        answer = [i['answer'] for i in batch]
        ques_inputs = self.tokenizer(question, max_length=question_length,
                                     padding='max_length', truncation=True, return_tensors='pt')

        with torch.no_grad():
            ques_embeds = self.text_embed_model(
                ques_inputs['input_ids'].to(device)).cpu().detach()
            image_inputs = self.feature_extractor(image, return_tensors="pt")
            for u, v in image_inputs.items():
                image_inputs[u] = v.to(device)
            img_embeds = self.img_model(
                **image_inputs).last_hidden_state.cpu().detach()
            inputs_embeds = torch.cat((ques_embeds, img_embeds), 1)

        clear_gpu()
        attention_mask = torch.cat((ques_inputs.attention_mask, torch.ones(
            img_embeds.shape[0], img_embeds.shape[1])), 1)
        del image_inputs, ques_embeds, img_embeds
        outputs = self.tokenizer(answer, max_length=answer_length,
                                 padding='max_length', truncation=True, return_tensors='pt')
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
        image = [i['image'] for i in batch]
        question = [i['lang'] + ': ' + i['question'] for i in batch]
#         question = [i['question'] for i in batch]
        answer = [i['answer'] for i in batch]
        ques_inputs = self.tokenizer(question, max_length=question_length,
                                     padding='max_length', truncation=True, return_tensors='pt')

        with torch.no_grad():
            ques_embeds = self.text_embed_model(
                ques_inputs['input_ids'].to(device)).cpu().detach()

            image_inputs = self.feature_extractor(image, return_tensors="pt")
            for u, v in image_inputs.items():
                image_inputs[u] = v.to(device)
            img_embeds = self.img_model(
                **image_inputs).last_hidden_state.cpu().detach()

            inputs_embeds = torch.cat((ques_embeds, img_embeds), 1)

        clear_gpu()
        attention_mask = torch.cat((ques_inputs.attention_mask, torch.ones(
            img_embeds.shape[0], img_embeds.shape[1])), 1)
        del image_inputs, ques_embeds, img_embeds

        result = {}
        result["inputs_embeds"] = inputs_embeds
        result["attention_mask"] = attention_mask

        return result


train_collator = DataCollator(
    vit_model, feature_extractor, tokenizer, embed_model)
test_collator = TestCollator(
    vit_model, feature_extractor, tokenizer, embed_model)

os.environ["NCCL_DEBUG"] = "INFO"

clear_gpu()

training_steps = int(len(train_ds) / (CFG.batch_size * CFG.grad_accm_steps))
eval_steps = int(training_steps / CFG.eval_in_epoch)
print(f'eval_steps: {eval_steps} | training_steps {training_steps}')

print('training:....')
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
    report_to='wandb',
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

clear_gpu()
trainer.train()
clear_gpu()
