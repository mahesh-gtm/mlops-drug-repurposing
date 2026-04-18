import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, GATv2Conv
from torch_geometric.data import Data
from src.utils.config import CONFIG
import os
import json
import mlflow
import mlflow.pytorch
from pathlib import Path
import pickle

class DrugRepurposingGNN(torch.nn.Module):
    def __init__(self, num_features=16, hidden_dim=CONFIG["hidden_dim"]):
        super().__init__()
        self.sage1 = GraphSAGE(num_features, hidden_dim, num_layers=2)
        self.sage2 = GraphSAGE(hidden_dim, hidden_dim, num_layers=2)
        # Predict edge weights: given two node embeddings, predict their connection strength
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
            # Removed Sigmoid to allow unbounded output for MSE loss
        )
    
    def forward(self, x, edge_index, edge_weight=None):
        """Get node embeddings"""
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        return x
    
    def predict_edge_strength(self, z, src_idx, dst_idx):
        """Predict connection strength between two nodes"""
        src_emb = z[src_idx]
        dst_emb = z[dst_idx]
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        # Output logits, will be converted to [0,1] range by training loss
        return self.mlp(combined).squeeze()

def train_gnn():
    print("🚀 Starting GNN training with drug-disease queries...")

    # === Ensure artifacts directory exists for MLflow ===
    artifacts_path = Path(CONFIG["artifacts_path"]).resolve()
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    # === MLflow Setup with absolute path ===
    mlflow_uri = Path(CONFIG["mlflow_tracking_uri"]).resolve()
    mlflow.set_tracking_uri(f"file:{str(mlflow_uri)}")
    mlflow.set_experiment(CONFIG["mlflow_experiment_name"])
    
    with mlflow.start_run(run_name="gnn-training") as run:
        print(f"📊 MLflow Run ID: {run.info.run_id}")
        
        # Log hyperparameters
        mlflow.log_param("num_epochs", CONFIG["num_epochs"])
        mlflow.log_param("learning_rate", CONFIG["learning_rate"])
        mlflow.log_param("hidden_dim", CONFIG["hidden_dim"])
        
        # Load the graph
        graph = torch.load("data/processed/graph.pt", weights_only=False)
        
        # Load mappings
        with open("data/processed/graph_mappings.pkl", 'rb') as f:
            mappings = pickle.load(f)
        
        node_mapping = mappings['node_mapping']
        drugs = mappings['drugs']
        diseases = mappings['diseases']

        model = DrugRepurposingGNN(num_features=graph.x.shape[1], hidden_dim=CONFIG["hidden_dim"])
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
        
        # Create drug-disease pairs for training
        print(f"🔄 Creating {len(drugs) * len(diseases)} drug-disease query pairs...")
        drug_disease_pairs = []
        for drug in drugs:
            for disease in diseases:
                drug_idx = node_mapping.get(drug)
                disease_idx = node_mapping.get(disease)
                if drug_idx is not None and disease_idx is not None:
                    drug_disease_pairs.append((drug_idx, disease_idx, drug, disease))
        
        print(f"✅ Created {len(drug_disease_pairs)} valid pairs")
        
        # Split into train/val/test
        num_pairs = len(drug_disease_pairs)
        train_size = int(CONFIG["train_split"] * num_pairs)
        val_size = int(CONFIG["val_split"] * num_pairs)
        
        train_pairs = drug_disease_pairs[:train_size]
        val_pairs = drug_disease_pairs[train_size:train_size + val_size]
        test_pairs = drug_disease_pairs[train_size + val_size:]
        
        print(f"📊 Pair splits: Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")
        
        # Create synthetic targets based on actual GDSC data
        # Drugs that work better on cell lines of a disease get higher scores
        import pandas as pd
        import networkx as nx
        
        gdsc_df = pd.read_csv("data/raw/gdsc2_original.csv")
        G = nx.Graph()
        for i in range(graph.num_nodes):
            G.add_node(i)
        for i in range(graph.edge_index.shape[1]):
            u = graph.edge_index[0, i].item()
            v = graph.edge_index[1, i].item()
            G.add_edge(u, v)
        
        def get_pair_score(drug_idx, disease_idx, drug_name, disease_name):
            """
            Score based on:
            1. GDSC efficacy data (actual IC50/AUC)
            2. Drug-specific multiplier (creates diversity)
            3. Disease-specific bias (creates diversity)
            """
            # Get cell lines for this disease
            cell_lines_for_disease = [c for c, d in mappings['cancer_mapping'].items() 
                                     if d == disease_name]
            
            # Find records for this drug on these cell lines
            mask = (gdsc_df['DRUG_NAME'] == drug_name) & (gdsc_df['CELL_LINE'].isin(cell_lines_for_disease))
            records = gdsc_df[mask]
            
            # Create drug-specific multiplier based on drug index - wider range for more diversity
            drug_multiplier = 0.3 + (drug_idx % len(drugs)) * 0.15  # Range: 0.3-0.95
            
            # Create disease-specific bias - wider range
            disease_bias = 0.05 + (disease_idx % len(diseases)) * 0.15  # Range: 0.05-0.7
            
            if len(records) > 0:
                # Average IC50 (lower is better) and AUC (higher is better)
                avg_ic50 = records['IC50'].mean()
                avg_auc = records['AUC'].mean()
                base_score = (1.0 / (1.0 + avg_ic50)) * avg_auc
                # Apply drug-specific multiplier and disease bias for diversity
                # Don't clip to 0.95 - allow wider range [0.0, 1.0]
                score = max(0.0, min(1.0, base_score * drug_multiplier + disease_bias))
            else:
                # Drug not tested - use drug-specific and disease-specific base
                score = max(0.0, min(1.0, drug_multiplier + disease_bias * 0.3))
            
            return torch.tensor(score, dtype=torch.float32)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(CONFIG["num_epochs"]):
            model.train()
            optimizer.zero_grad()
            
            # Get node embeddings
            z = model(graph.x, graph.edge_index, graph.edge_weight)
            
            # Train on drug-disease pairs - accumulate loss
            train_loss = 0
            for drug_idx, disease_idx, drug_name, disease_name in train_pairs:
                target = get_pair_score(drug_idx, disease_idx, drug_name, disease_name)
                pred = model.predict_edge_strength(z, drug_idx, disease_idx)
                # Apply sigmoid to convert unbounded output to [0, 1], matching target scale
                pred_sigmoid = torch.sigmoid(pred)
                loss = F.mse_loss(pred_sigmoid.unsqueeze(0), target.unsqueeze(0))
                train_loss += loss
            
            # Average loss and backward once
            train_loss = train_loss / len(train_pairs) if train_pairs else train_loss
            train_loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                z_val = model(graph.x, graph.edge_index, graph.edge_weight)
                val_loss = 0
                for drug_idx, disease_idx, drug_name, disease_name in val_pairs:
                    target = get_pair_score(drug_idx, disease_idx, drug_name, disease_name)
                    pred = model.predict_edge_strength(z_val, drug_idx, disease_idx)
                    # Apply sigmoid to convert unbounded output to [0, 1]
                    pred_sigmoid = torch.sigmoid(pred)
                    loss = F.mse_loss(pred_sigmoid.unsqueeze(0), target.unsqueeze(0))
                    val_loss += loss
                val_loss = val_loss / len(val_pairs) if val_pairs else torch.tensor(0.0)
            
            if epoch % CONFIG["log_metrics_every"] == 0:
                print(f"Epoch {epoch:3d} - Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
                mlflow.log_metrics({
                    "train_loss": train_loss.item(),
                    "val_loss": val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
                }, step=epoch)
            
            # Early stopping
            val_loss_scalar = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
            if val_loss_scalar < best_val_loss - CONFIG["early_stopping_threshold"]:
                best_val_loss = val_loss_scalar
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"⏹️  Early stopping at epoch {epoch}")
                break
        
        # === Evaluation on Test Set ===
        model.eval()
        with torch.no_grad():
            z_test = model(graph.x, graph.edge_index, graph.edge_weight)
            test_loss = 0
            predictions = []
            targets = []
            
            for drug_idx, disease_idx, drug_name, disease_name in test_pairs:
                target = get_pair_score(drug_idx, disease_idx, drug_name, disease_name)
                pred = model.predict_edge_strength(z_test, drug_idx, disease_idx)
                # Apply sigmoid to convert unbounded output to [0, 1]
                pred_sigmoid = torch.sigmoid(pred)
                loss = F.mse_loss(pred_sigmoid.unsqueeze(0), target.unsqueeze(0))
                test_loss += loss
                predictions.append(pred_sigmoid.item())
                targets.append(target.item())
            
            test_loss = test_loss / len(test_pairs) if test_pairs else torch.tensor(0.0)
            
            # Compute metrics
            predictions = torch.tensor(predictions)
            targets = torch.tensor(targets)
            mae = F.l1_loss(predictions, targets)
            
            # R² score
            ss_res = ((predictions - targets) ** 2).sum()
            ss_tot = ((targets - targets.mean()) ** 2).sum()
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else torch.tensor(0.0)
        
        print(f"\n📈 Final Evaluation Metrics:")
        print(f"   Test MSE Loss: {test_loss.item():.4f}")
        print(f"   Test MAE: {mae.item():.4f}")
        print(f"   Test R² Score: {r2_score.item():.4f}")
        print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"   Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        
        # Log test metrics
        mlflow.log_metrics({
            "test_loss": test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss,
            "test_mae": mae.item(),
            "test_r2_score": r2_score.item(),
            "best_val_loss": best_val_loss,
            "final_train_loss": train_loss.item(),
            "final_val_loss": val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss,
            "pred_min": predictions.min().item(),
            "pred_max": predictions.max().item(),
            "target_min": targets.min().item(),
            "target_max": targets.max().item(),
        })

        # Save model
        artifacts_path = Path(CONFIG["artifacts_path"]).resolve()
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        model_path = artifacts_path / "model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"✅ GNN model saved to {model_path}")
        
        # Log model to MLflow (simplified: just log artifact, skip pytorch.log_model due to permissions)
        try:
            mlflow.log_artifact(str(model_path))
        except Exception as e:
            print(f"⚠️  Warning: Could not log model artifact to MLflow: {e}")

        # Save comprehensive metrics
        metrics = {
            "final_train_loss": float(train_loss.item()),
            "final_val_loss": float(val_loss.item()),
            "test_loss": float(test_loss.item()),
            "test_mae": float(mae.item()),
            "test_r2_score": float(r2_score.item()),
            "epochs_trained": epoch + 1,
            "early_stopped": patience_counter >= CONFIG["early_stopping_patience"],
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "num_drugs": len(drugs),
            "num_diseases": len(diseases),
        }
        
        metrics_path = artifacts_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        try:
            mlflow.log_artifact(str(metrics_path))
        except Exception as e:
            print(f"⚠️  Warning: Could not log metrics artifact to MLflow: {e}")
        
        print(f"💾 Metrics saved to {metrics_path}")
        print(f"📊 View results: mlflow ui --backend-store-uri {CONFIG['mlflow_tracking_uri']}")
        print("🎉 Training completed successfully!")

if __name__ == "__main__":
    train_gnn()