import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # Data paths
    "raw_data_path": "data/raw/",
    "processed_data_path": "data/processed/",
    "split_data_path": "data/splits/",
    "artifacts_path": "artifacts/",
    
    # MLflow configuration
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "./mlruns"),
    "mlflow_experiment_name": "drug-repurposing-gnn",
    
    # Training configuration
    "random_seed": 42,
    "num_epochs": 150,
    "learning_rate": 0.02,  # Balanced rate between 0.01 and 0.05
    "hidden_dim": 96,  # Balanced between 64 and 128
    "batch_size": 32,
    
    # Data split configuration
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
    
    # Monitoring configuration
    "log_metrics_every": 10,
    "early_stopping_patience": 20,
    "early_stopping_threshold": 0.001,
}