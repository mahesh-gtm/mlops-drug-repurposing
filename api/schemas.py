from pydantic import BaseModel

class RepurposingRequest(BaseModel):
    drug_id: str
    target_disease: str

class RepurposingResponse(BaseModel):
    drug_id: str
    target_disease: str
    repurposing_score: float
    confidence: str
    explanation: str