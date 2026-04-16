import torch
import pickle
from pathlib import Path
import pandas as pd
from src.training.train_gnn import DrugRepurposingGNN
from src.utils.config import CONFIG

model = None
graph = None
mappings = None
gdsc_data = None

def load_model_and_graph():
    global model, graph, mappings, gdsc_data
    artifacts_path = Path(CONFIG["artifacts_path"])
    processed_path = Path(CONFIG["processed_data_path"])
    
    model_path = artifacts_path / "model.pt"
    graph_path = processed_path / "graph.pt"
    mappings_path = processed_path / "graph_mappings.pkl"

    try:
        # Load graph
        graph = torch.load(graph_path, weights_only=False)
        
        # Load mappings
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        # Load GDSC data
        gdsc_data = pd.read_csv("data/raw/gdsc2_original.csv")
        
        # Load model
        model = DrugRepurposingGNN(
            num_features=graph.x.shape[1],
            hidden_dim=CONFIG["hidden_dim"]
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        print("✅ Real trained GNN model loaded successfully for inference.")
        print(f"   Drugs: {len(mappings['drugs'])}")
        print(f"   Diseases: {len(mappings['diseases'])}")
        print(f"   Cell Lines: {len(mappings['cell_lines'])}")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
        import traceback
        traceback.print_exc()

load_model_and_graph()

def predict_repurposing(drug_id: str, target_disease: str):
    """Predict drug-disease connection strength using GNN"""
    
    if model is None or graph is None or mappings is None:
        return {
            "drug_id": drug_id,
            "target_disease": target_disease,
            "repurposing_score": 0.5,
            "confidence": "low",
            "explanation": "Model not loaded yet.",
            "error": True
        }

    try:
        node_mapping = mappings['node_mapping']
        drugs = mappings['drugs']
        diseases = mappings['diseases']
        cancer_mapping = mappings['cancer_mapping']
        
        # Find drug and disease indices (case-insensitive)
        drug_idx = None
        disease_idx = None
        actual_drug_name = None
        actual_disease_name = None
        
        if drug_id in node_mapping:
            drug_idx = node_mapping[drug_id]
            actual_drug_name = drug_id
        else:
            for drug in drugs:
                if drug.lower() == drug_id.lower():
                    drug_idx = node_mapping[drug]
                    actual_drug_name = drug
                    break
        
        if target_disease in node_mapping:
            disease_idx = node_mapping[target_disease]
            actual_disease_name = target_disease
        else:
            for disease in diseases:
                if disease.lower() == target_disease.lower():
                    disease_idx = node_mapping[disease]
                    actual_disease_name = disease
                    break
        
        if drug_idx is None or disease_idx is None:
            return {
                "drug_id": drug_id,
                "target_disease": target_disease,
                "repurposing_score": 0.0,
                "confidence": "low",
                "explanation": "Drug or disease not found.",
                "error": True
            }
        
        # Get GNN prediction
        model.eval()
        with torch.no_grad():
            z = model(graph.x, graph.edge_index, graph.edge_weight if hasattr(graph, 'edge_weight') else None)
            nn_score = model.predict_edge_strength(z, drug_idx, disease_idx).item()
            # Apply sigmoid to convert unbounded output to [0, 1] range
            nn_score = torch.sigmoid(torch.tensor(nn_score)).item()
        
        # Get GDSC-based score
        cell_lines_for_disease = [c for c, d in cancer_mapping.items() if d == actual_disease_name]
        mask = (gdsc_data['DRUG_NAME'] == actual_drug_name) & (gdsc_data['CELL_LINE'].isin(cell_lines_for_disease))
        records = gdsc_data[mask]
        
        if len(records) > 0:
            avg_ic50 = records['IC50'].mean()
            avg_auc = records['AUC'].mean()
            gdsc_score = (1.0 / (1.0 + avg_ic50)) * avg_auc
            explanation = f"GNN: {nn_score:.3f} | GDSC: IC50={avg_ic50:.2f}, AUC={avg_auc:.2f}"
        else:
            gdsc_score = 0.0
            explanation = f"GNN: {nn_score:.3f} | Not tested on {actual_disease_name}"
        
        # Combine scores
        final_score = 0.7 * nn_score + 0.3 * gdsc_score if gdsc_score > 0 else nn_score
        
        confidence = "high" if final_score > 0.7 else "medium" if final_score > 0.4 else "low"
        
        return {
            "drug_id": actual_drug_name,
            "target_disease": actual_disease_name,
            "repurposing_score": round(final_score, 3),
            "confidence": confidence,
            "explanation": explanation,
            "error": False
        }

    except Exception as e:
        return {
            "drug_id": drug_id,
            "target_disease": target_disease,
            "repurposing_score": 0.0,
            "confidence": "error",
            "explanation": f"Error: {str(e)}",
            "error": True
        }
