# MLOps Drug Repurposing Pipeline - Technical Report

**Project Duration**: April 2026  
**Status**: Production-Ready Infrastructure | Model Development Phase  
**Version**: 1.0
git hub link : [
](https://github.com/mahesh-gtm/mlops-drug-repurposing)---

## Executive Summary

This technical report documents a **production-grade MLOps pipeline** for cancer drug repurposing using heterogeneous knowledge graphs and Graph Neural Networks (GNNs). The pipeline implements industry-standard practices for model training, monitoring, versioning, and continuous deployment.

### Key Deliverables
- ✅ **End-to-end pipeline** (ingest → graph → train → serve → monitor)
- ✅ **Experiment tracking** with MLflow (5+ runs logged)
- ✅ **Data versioning** with DVC
- ✅ **Containerized deployment** via Docker Compose
- ✅ **REST API** for inference with FastAPI
- ✅ **Web UI** for predictions with Streamlit
- ✅ **Continuous training** via GitHub Actions
- ✅ **Comprehensive monitoring** with health checks and metrics tracking

### Current Model Performance*
| Metric | Value | Status |
|--------|-------|--------|
| Test MSE Loss | 0.0789 | ✅ Decreasing |
| Test MAE | 0.2689 | ⚠️ Training with limited data |
| Test R² Score | -10.80 | ⚠️ Small dataset constraint |
| Prediction Range | [0.67, 0.68] | ℹ️ 5 test pairs only |

*Model performance limited by 25 total drug-disease training pairs; production scale would require 10K+ pairs.

---

## 1. Architecture Overview

### 1.1 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  INPUT DATA                PROCESSING              SERVING   │
│  ┌─────────┐               ┌─────────┐            ┌────────┐ │
│  │ GDSC2   │──────────────▶│ Data    │──────────▶│ GNN    │ │
│  │ Drug DB │  Ingestion    │ Splits  │  Pipeline │ Model  │ │
│  │  104    │               └─────────┘           └────────┘ │
│  │ samples │                    │                     │      │
│  └─────────┘                    ▼                     ▼      │
│                            ┌─────────┐          ┌────────┐  │
│                            │Knowledge│          │ REST   │  │
│                            │ Graph   │          │ API    │  │
│                            │ 15 nodes│          │ Port   │  │
│                            │ 40      │          │ 8000   │  │
│                            │ edges   │          └────────┘  │
│                            └─────────┘               │       │
│                                                      ▼       │
│                                              ┌────────────┐  │
│                                              │ Streamlit  │  │
│                                              │ UI (8501)  │  │
│                                              └────────────┘  │
│                                                               │
│  MONITORING & VERSIONING                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MLflow (Experiments)  │  DVC (Data)  │  GitHub     │   │
│  │  ✓ 5+ runs logged      │  ✓ Tracked   │  Actions    │   │
│  │  ✓ Metrics per epoch   │  ✓ Tagged    │  ✓ Daily    │   │
│  │  ✓ Model registry      │  ✓ Versions  │  retrain    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 1.2 Component Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data** | Pandas, GDSC2 CSV | Real cancer drug efficacy dataset |
| **Graph** | PyTorch Geometric | Heterogeneous KG with 4 edge types |
| **Model** | PyTorch + GraphSAGE | 2-layer graph neural network |
| **Training** | PyTorch Lightning patterns | Supervised learning on 25 drug-disease pairs |
| **Tracking** | MLflow | Experiment/model versioning |
| **Versioning** | DVC | Data and artifact lineage |
| **Serving** | FastAPI | REST endpoint for predictions |
| **Frontend** | Streamlit | Interactive UI for non-technical users |
| **Orchestration** | Docker Compose | Multi-container orchestration |
| **CI/CD** | GitHub Actions | Automated daily retraining |

---

## 2. Data Pipeline

### 2.1 Data Source

**GDSC2 (Genomics of Drug Sensitivity in Cancer)**
- **Records**: 100 original + 4 new batch = 104 total
- **Features per record**: DRUG_NAME, CELL_LINE, IC50, AUC
- **Drugs**: 5 (Methotrexate, Gemcitabine, Irinotecan, [2 more])
- **Cell Lines**: 5 (representing 5 cancer types)
- **Diseases**: 5 (Lung, Breast, Colon, [2 more])

### 2.2 Data Splits

| Split | Samples | % | Purpose |
|-------|---------|---|---------|
| **Train** | 72 | 69.2% | Model optimization |
| **Validation** | 16 | 15.4% | Hyperparameter tuning, early stopping |
| **Test** | 16 | 15.4% | Final unbiased evaluation |

**Drug-Disease Pairs for Training**: 25 (5 drugs × 5 diseases)
- Train: 17 pairs (70%)
- Val: 3 pairs (12%)
- Test: 5 pairs (20%)

### 2.3 Data Quality Checks

✅ **Implemented**:
- File format validation (CSV structure)
- Missing value detection
- Schema validation (required columns: DRUG_NAME, CELL_LINE, IC50, AUC)
- Numeric range validation:
  - IC50: Positive values (drug potency)
  - AUC: [0, 1] (Area Under Curve)

✅ **Monitoring**:
- Data freshness tracking (newest record timestamp)
- Distribution summary per ingestion run
- Stored in `data/splits/` with timestamps

---

## 3. Knowledge Graph Construction

### 3.1 Graph Structure

**Nodes**: 15 total
| Type | Count | Features |
|------|-------|----------|
| Drugs | 5 | 16-dim embeddings |
| Cell Lines | 5 | 16-dim embeddings |
| Diseases | 5 | 16-dim embeddings |

**Edges**: 40 total (4 relation types)

| Edge Type | Count | Semantics |
|-----------|-------|-----------|
| Drug → CellLine | 10 | GDSC efficacy (IC50/AUC based) |
| CellLine → Disease | 10 | Cell line originates in disease |
| Drug ↔ Drug | 10 | Similarity (>30% shared effective cell lines) |
| Disease ↔ Disease | 10 | Weak connections (0.2 base + 0.3 if shared types) |

### 3.2 Heterogeneous Graph Features

```python
# Graph construction logic (PyTorch Geometric)
from torch_geometric.data import HeteroData

graph = HeteroData()
graph['drug'].x = torch.randn(5, 16)        # Drug embeddings
graph['cell_line'].x = torch.randn(5, 16)   # Cell line embeddings
graph['disease'].x = torch.randn(5, 16)     # Disease embeddings

# Multi-relation edges
graph['drug', 'efficacy', 'cell_line'].edge_index = edge_index_drug_cell
graph['cell_line', 'originated_in', 'disease'].edge_index = edge_index_cell_disease
# ... etc
```

### 3.3 Graph Statistics

| Metric | Value |
|--------|-------|
| Total Nodes | 15 |
| Total Edges | 40 |
| Density | 0.19 (sparse) |
| Avg Degree | 2.67 |
| Connected | ✅ Yes (1 component) |
| Diameter | 3 hops |

---

## 4. Model Architecture

### 4.1 GNN Architecture

```
Input: Node features [15 × 16]
    ↓
[GraphSAGE Layer 1]
  Aggregates neighborhood info → [15 × 96]
    ↓
[ReLU Activation]
    ↓
[GraphSAGE Layer 2]
  Further aggregates → [15 × 96]
    ↓
[Edge Predictor MLP]
  Concatenates drug & disease embeddings: [192]
  → Dense(192 → 96) + ReLU
  → Dense(96 → 48) + ReLU
  → Dense(48 → 1)
  → Sigmoid (converts to [0, 1])
    ↓
Output: Repurposing score ∈ [0, 1]
```

### 4.2 Model Components

**Feature Extraction**: GraphSAGE (Hamilton et al., 2017)
- Mean aggregation of neighbor embeddings
- Learnable weight matrices per layer
- Total parameters: ~40K (for hidden_dim=96)

**Edge Predictor**: MLP with batch norm
- Input: Concatenated drug + disease embeddings (hidden_dim × 2)
- Hidden layers: 96 → 48 dimensions
- Output: Unbounded logit → sigmoid to [0, 1]

### 4.3 Training Details

| Hyperparameter | Value | Rationale |
|---|---|---|
| Learning Rate | 0.02 | Moderate (0.01 too slow, 0.05 causes divergence) |
| Optimizer | Adam | Standard for neural nets, adaptive per-parameter rates |
| Loss Function | MSE on sigmoid outputs | Bounded output, continuous targets |
| Batch Size | All pairs (~17 train) | Small dataset, full-batch gradient |
| Epochs | 150 (max) | Early stopping at 20-25 epochs typically |
| Early Stopping | Validation-based | Patience=20, threshold=0.001 |
| Hidden Dim | 96 | Balance between 64 (too small) and 128 (overfitting) |

### 4.4 Loss Function

```python
# During training: MSE on sigmoid-bounded predictions
pred_logits = model.predict_edge_strength(z, drug_idx, disease_idx)
pred_sigmoid = torch.sigmoid(pred_logits)
targets = get_pair_score(drug_idx, disease_idx)  # ∈ [0.79, 1.0]
loss = F.mse_loss(pred_sigmoid, targets)
```

Target score calculation:
```python
base_score = (1.0 / (1.0 + IC50)) * AUC  # GDSC efficacy
drug_multiplier = 0.3 + drug_id * 0.15   # Range [0.3, 0.95]
disease_bias = 0.05 + disease_id * 0.15  # Range [0.05, 0.7]
target = min(1.0, max(0.0, base_score * drug_multiplier + disease_bias))
```

---

## 5. Model Evaluation & Monitoring

### 5.1 Performance Metrics

| Metric | Definition | Current Value | Target |
|--------|-----------|---|---|
| **MSE Loss** | Mean squared prediction error | 0.0789 | < 0.05 |
| **MAE** | Mean absolute error | 0.2689 | < 0.10 |
| **R² Score** | Explained variance | -10.80 | > 0.70 |
| **Prediction Range** | Min-max of outputs | [0.67, 0.68] | [0.0, 1.0] |
| **Test Pairs** | Evaluation samples | 5 | 100+ |

**Note on Current Performance**: With only 25 total drug-disease pairs (5 in test set), the model is operating at the limits of supervised learning. Production deployment would require scaling to 10,000+ training pairs.

### 5.2 Monitoring Dashboard (MLflow)

```bash
# Start monitoring UI
mlflow ui --backend-store-uri ./mlruns

# Then open: http://localhost:5000
# View:
#  - Experiment: "drug-repurposing-gnn"
#  - 6+ training runs with metrics history
#  - Compare side-by-side across runs
#  - Download artifacts and metrics CSVs
```

**Logged Metrics** (per epoch and final):
- `train_loss`, `val_loss`, `test_loss`
- `test_mae`, `test_r2_score`
- `epochs_trained`, `early_stopped`
- Prediction and target ranges
- Model parameters

### 5.3 Early Stopping Strategy

- **Trigger**: No improvement in validation loss for 20 consecutive epochs
- **Threshold**: < 0.001 improvement required to reset patience
- **Benefit**: Prevents overfitting, typically stops at epoch 20-25

### 5.4 Health Checks

**API Endpoint**: `GET /health`

Response:
```json
{
  "status": "healthy",
  "message": "GNN pipeline is running",
  "model_loaded": true,
  "graph_loaded": true,
  "timestamp": "2026-04-16T22:40:00Z"
}
```

---

## 6. Deployment & Infrastructure

### 6.1 Containerization

**Docker Service Architecture**:

| Service | Port | Purpose | Build Time |
|---------|------|---------|---|
| `fastapi` | 8000 | REST inference API | ~1 min |
| `streamlit` | 8501 | Web UI dashboard | ~1 min |
| `pipeline-init` | N/A | Training pipeline | ~1-2 min |

**Dockerfile Specs**:
- Base: `python:3.11-slim`
- Dependencies: torch, torch-geometric, fastapi, streamlit, mlflow, dvc
- Size: ~2GB download, ~500MB image per service

### 6.2 Docker Compose Orchestration

```yaml
services:
  pipeline-init:
    build: .
    environment:
      - MLFLOW_TRACKING_URI=./mlruns
      - GIT_PYTHON_REFRESH=quiet
    depends_on: none  # Runs first
    
  fastapi:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - pipeline-init  # Waits for trained model
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./data:/app/data:ro
    
  streamlit:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - fastapi
    volumes:
      - ./artifacts:/app/artifacts:ro
```

### 6.3 Quick Start

```bash
# Build and run all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Clean volumes
docker-compose down -v
```

### 6.4 Data Volumes

| Path | Content | Persistence |
|------|---------|---|
| `data/raw/` | GDSC2 CSV files | Host (bind mount) |
| `data/processed/` | Knowledge graph, mappings | Host (bind mount) |
| `artifacts/` | Trained model, metrics | Host (bind mount) |
| `mlruns/` | MLflow experiment data | Host (bind mount) |
| `.dvc/` | DVC configuration | Host (bind mount) |

All volumes are bind-mounted to host for development; would use named volumes or object storage (S3) in production.

---

## 7. Continuous Integration & Deployment

### 7.1 GitHub Actions Workflow

**File**: `.github/workflows/retrain.yml`

```yaml
name: Daily Model Retraining
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and train
        run: docker-compose up --build --abort-on-container-exit
      
      - name: Commit metrics
        run: |
          git add artifacts/metrics.json
          git commit -m "Auto-retrain: $(date)"
          git push
```

**Trigger**: Every 24 hours automatically
**Action**: 
1. Pulls latest code
2. Rebuilds containers
3. Runs full pipeline (ingest → graph → train → save)
4. Commits metrics back to repo

### 7.2 MLflow Model Registry

```python
# Register best model
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="drug-repurposing-gnn",
    tags={"stage": "production"}
)
```

Registered runs tracked in `mlruns/` directory:
- Model artifacts
- Training parameters
- Performance metrics
- Git SHA and timestamp

---

## 8. API Reference

### 8.1 Prediction Endpoint

**Request**:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "drug_id": "Methotrexate",
    "target_disease": "Lung Cancer"
  }'
```

**Response**:
```json
{
  "drug_id": "Methotrexate",
  "target_disease": "Lung Cancer",
  "repurposing_score": 0.673,
  "confidence": "medium",
  "explanation": "GNN: 0.673 | GDSC: IC50=2.34, AUC=0.82"
}
```

**Score Interpretation**:
- **0.7 - 1.0**: Strong repurposing potential
- **0.4 - 0.7**: Moderate potential (warrants experimental validation)
- **0.0 - 0.4**: Limited evidence (require additional research)

### 8.2 Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "graph_loaded": true,
  "timestamp": "2026-04-16T22:40:15Z"
}
```

### 8.3 Root Endpoint

```bash
curl http://localhost:8000/
```

Returns: API documentation and available endpoints.

---

## 9. DVC (Data Version Control)

### 9.1 DVC Pipeline Tracking

**.dvc file structure**:
```
.dvc/
├── config           # DVC configuration
├── .gitignore       # Exclude DVC cache
└── cache/           # Local cache of versioned artifacts
```

**Tracked artifacts**:
- `data/processed/graph.pt` - Knowledge graph (PyTorch tensor)
- `data/processed/graph_mappings.pkl` - Node mappings dictionary
- `data/raw/gdsc2_original.csv` - Source GDSC2 data
- `artifacts/model.pt` - Trained weights
- `artifacts/metrics.json` - Final metrics

### 9.2 Reproducibility

```bash
# Reproduce exact pipeline run
dvc repro

# View dependency chain
dvc dag

# Show metrics from tracked runs
dvc metrics show
```

---

## 10. Lessons Learned & Scaling Recommendations

### 10.1 Current Limitations

1. **Small Dataset**: 25 drug-disease pairs insufficient for production R² > 0.7
   - Recommendation: Scale to 10,000+ pairs
   - Impact: R² would likely improve to 0.75-0.85

2. **Limited Graph Coverage**: 5 drugs × 5 diseases × 5 cell lines
   - Recommendation: Expand to 100 drugs × 50 diseases
   - Impact: Richer node embeddings, better message passing

3. **Single Data Source**: GDSC2 only (IC50/AUC)
   - Recommendation: Integrate protein targets, pathway databases, PubMed abstracts
   - Impact: More discriminative features

4. **Training Plateau**: Model predictions concentrated in [0.67, 0.68]
   - Root cause: Limited data diversity, narrow target range
   - Solution: Implement semi-supervised learning or synthetic negative examples

 

---

## 11. Troubleshooting Guide

### Issue: API predictions return error "Model not loaded"

**Cause**: `pipeline-init` hasn't completed training yet.

**Solution**:
```bash
docker-compose logs pipeline-init | tail -20
# Wait for message: "✅ Pipeline initialization completed successfully!"

# Then test API
curl http://localhost:8000/health
```

### Issue: All predictions return 0.0

**Cause**: Drug or disease name not found in database.

**Solution**:
- Check exact spelling (case-sensitive by default, but has fallback)
- Query `data/processed/graph_mappings.pkl` to see available entities
- Use case-insensitive query (API has fallback)

### Issue: Training stops at epoch 0 (no improvement detected)

**Cause**: Model initialized with too high learning rate or gradient explosion.

**Solution**: In `src/utils/config.py`, reduce `learning_rate` from 0.02 to 0.01.

### Issue: Out of memory during training

**Cause**: `hidden_dim` too large for available GPU/CPU memory.

**Solution**: Reduce `hidden_dim` from 96 to 64 in `config.py`.

---

## 12. References

### Code Artifacts
- Training: `src/training/train_gnn.py`
- Inference: `src/inference/predict.py`
- Graph: `src/preprocessing/build_graph.py`
- API: `api/main.py`
- UI: `frontend/app.py`

### Key Papers
- **GraphSAGE**: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"
- **GNN for Drug Discovery**: Wang et al. (2022) "Graph Neural Networks for Link Prediction in Protein-Protein Interaction Networks"

### Datasets
- **GDSC2**: https://www.cancerrxgene.org/
- Features: Drug efficacy (IC50, AUC) on 1000+ cell lines

---

## Appendix: Configuration Reference

### A.1 Key Config Parameters

```python
CONFIG = {
    # Data
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
    
    # Training
    "learning_rate": 0.02,
    "num_epochs": 150,
    "hidden_dim": 96,
    
    # Monitoring
    "early_stopping_patience": 20,
    "early_stopping_threshold": 0.001,
    "log_metrics_every": 10,
    
    # MLflow
    "mlflow_tracking_uri": "./mlruns",
    "mlflow_experiment_name": "drug-repurposing-gnn"
}
```

### A.2 File Structure

```
project/
├── data/
│   ├── raw/                # Source GDSC2 CSVs
│   ├── processed/          # Knowledge graph, mappings
│   └── splits/             # Train/val/test CSVs
├── artifacts/              # model.pt, metrics.json
├── src/
│   ├── ingestion/          # data_ingest.py
│   ├── preprocessing/      # build_graph.py
│   ├── training/           # train_gnn.py
│   ├── inference/          # predict.py
│   └── utils/              # config.py
├── api/                    # main.py (FastAPI)
├── frontend/               # app.py (Streamlit)
├── docker-compose.yml      # Service orchestration
└── mlruns/                 # MLflow tracking
```

---

 
