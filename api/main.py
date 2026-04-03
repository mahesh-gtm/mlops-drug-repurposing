from fastapi import FastAPI
from src.inference.predict import predict_repurposing
from api.schemas import RepurposingRequest, RepurposingResponse

app = FastAPI(title="Drug Repurposing GNN API")

@app.post("/predict", response_model=RepurposingResponse)
def predict(request: RepurposingRequest):
    result = predict_repurposing(request.drug_id, request.target_disease)
    return result

@app.get("/health")
def health():
    return {"status": "healthy"}