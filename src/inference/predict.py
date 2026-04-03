import torch
from src.utils.config import CONFIG

def predict_repurposing(drug_id: str, target_disease: str):
    # Simple mock prediction for demo (replace with real GNN inference later)
    model_score = 0.87 if drug_id == "DRUG1" else 0.62
    explanation = f"Strong link via GENE_A → {target_disease} pathway"
    
    return {
        "drug_id": drug_id,
        "target_disease": target_disease,
        "repurposing_score": float(model_score),
        "confidence": "high",
        "explanation": explanation
    }