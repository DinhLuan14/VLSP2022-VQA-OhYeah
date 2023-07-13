# config.py

from dataclasses import dataclass


@dataclass
class Config:
    fold: int = 0
    seed: int = 42

    kfold: int = 10
    epochs: int = 10
    batch_size: int = 64

    lr: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    max_len: int = 128
    vocab_size: int = 30000

    model_name: str = "facebook/m2m100_418M"

    train_file: str = "data/train.csv"
    test_file: str = "data/test.csv"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()
