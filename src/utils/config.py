import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "artifacts_path": "artifacts/",
    "mlflow_tracking_uri": "http://localhost:5000",
    "model_name": "gnn_drug_repurposing",
    "random_seed": 42,
    "num_epochs": 50,
    "learning_rate": 0.01,
    "hidden_dim": 64,
}