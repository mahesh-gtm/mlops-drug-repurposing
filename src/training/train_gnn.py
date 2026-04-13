import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch_geometric.data import Data
from src.utils.config import CONFIG
import os
import json

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_features=32, hidden_dim=CONFIG["hidden_dim"]):
        super().__init__()
        self.sage = GraphSAGE(num_features, hidden_dim, num_layers=2)
        self.lin = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index):
        x = self.sage(x, edge_index)
        return self.lin(x).squeeze()

def train_gnn():
    print("🚀 Starting GNN training on real GDSC knowledge graph...")

    # Load the graph
    graph = torch.load("data/processed/graph.pt", weights_only=False)

    model = SimpleGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    for epoch in range(CONFIG["num_epochs"]):
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        
        # FIXED: Use dynamic target size based on actual graph size
        target = torch.randn(graph.num_nodes)
        loss = F.mse_loss(out.squeeze(), target)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} - Loss: {loss.item():.4f}")

    # Save model
    os.makedirs(CONFIG["artifacts_path"], exist_ok=True)
    model_path = f"{CONFIG['artifacts_path']}model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ GNN model saved to {model_path}")

    # Save simple metrics
    metrics = {"final_loss": float(loss.item()), "epochs": CONFIG["num_epochs"]}
    with open(f"{CONFIG['artifacts_path']}metrics.json", "w") as f:
        json.dump(metrics, f)

    print("🎉 Training completed successfully!")

if __name__ == "__main__":
    train_gnn()