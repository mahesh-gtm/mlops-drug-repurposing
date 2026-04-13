import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "artifacts_path": "artifacts/",
    "random_seed": 42,
    "num_epochs": 50,
    "learning_rate": 0.01,
    "hidden_dim": 64,
}