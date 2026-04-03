import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data
from src.utils.config import CONFIG
import mlflow
import json
import os

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features=16, hidden_dim=CONFIG["hidden_dim"]):
        super().__init__()
        self.sage = GraphSAGE(num_features, hidden_dim, num_layers=2)
        self.lin = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        return self.lin(x).squeeze()

def train_gnn():
    mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
    mlflow.set_experiment("drug_repurposing_gnn")
    
    with mlflow.start_run(run_name="gnn_training_run"):
        mlflow.log_params({
            "epochs": CONFIG["num_epochs"],
            "lr": CONFIG["learning_rate"],
            "hidden_dim": CONFIG["hidden_dim"]
        })
        
        # Load graph
        graph = torch.load("data/processed/graph.pt")
        
        # Dummy node features for exam
        x = torch.randn(graph.num_nodes, 16)
        
        model = SimpleGNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        
        for epoch in range(CONFIG["num_epochs"]):
            optimizer.zero_grad()
            out = model(x, graph.edge_index)
            loss = F.mse_loss(out[:3], torch.tensor([0.8, 0.3, 0.9]))  # dummy target
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                mlflow.log_metric("loss", loss.item(), step=epoch)
                print(f"Epoch {epoch} - Loss: {loss.item():.4f}")
        
        # Save model
        os.makedirs(CONFIG["artifacts_path"], exist_ok=True)
        torch.save(model.state_dict(), f"{CONFIG['artifacts_path']}model.pt")
        
        mlflow.log_artifact(f"{CONFIG['artifacts_path']}model.pt")
        mlflow.log_metric("final_loss", loss.item())
        
        print("✅ Model trained and logged to MLflow.")

if __name__ == "__main__":
    train_gnn()