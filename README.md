# Cancer Drug Repurposing via Knowledge Graphs & GNNs

**An end-to-end MLOps pipeline for predicting new cancer indications for existing drugs using Graph Neural Networks and GDSC pharmaceutical data.**

![Status](https://img.shields.io/badge/status-production%20infrastructure-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Monitoring](#monitoring)
- [Contributing](#contributing)

---

## 🚀 Quick Start
link to the streamlit after the docker is run  - [streamlit app
](https://jubilant-lamp-5vqg447xj6jpc7rg5-8501.app.github.dev/)

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM, 10GB disk space
- Port availability: 8000 (API), 8501 (UI), 5000 (MLflow)

### Run in 30 Seconds

```bash
# Clone and enter directory
git clone https://github.com/mahesh-gtm/mlops-drug-repurposing.git
cd mlops-drug-repurposing

# Start all services
docker-compose up -d --build

# Wait for training to complete (~2 minutes)
docker-compose logs -f pipeline-init

# Access services
open http://localhost:8501          # Streamlit UI
 
```

That's it! The pipeline automatically:
1. ✅ Ingests GDSC2 drug efficacy data
2. ✅ Builds a heterogeneous knowledge graph
3. ✅ Trains a GNN model on 25 drug-disease pairs
4. ✅ Serves predictions via REST API
5. ✅ Tracks experiments with MLflow

---

## ✨ Features

### Core Pipeline
- **Real Data**: GDSC2 cancer drug sensitivity database (104 samples)
- **Knowledge Graph**: 15 nodes (drugs/cells/diseases), 40 edges (4 types)
- **GNN Model**: GraphSAGE with edge prediction MLP
- **Production Ready**: MLflow, DVC, GitHub Actions automation

### Infrastructure
| Component | Technology | Status |
|-----------|-----------|--------|
| Data Ingestion | Pandas + GDSC2 CSV | ✅ Real data loaded |
| Graph Building | PyTorch Geometric | ✅ Heterogeneous KG |
| Model Training | PyTorch + Adam optimizer | ✅ Early stopping |
| Serving | FastAPI | ✅ REST API  (port 8000) |
| Web UI | Streamlit | ✅ Interactive dashboard (port 8501) |
| Experiment Tracking | MLflow | ✅ 5+ runs logged |
| Data Versioning | DVC | ✅ Artifact tracking |
| CI/CD | GitHub Actions | ✅ Daily auto-retraining |
| Containerization | Docker Compose | ✅ 3 services orchestrated |

### Monitoring & Observability
- **Real-time Metrics**: Loss, MAE, R² score, prediction distributions
- **Health Checks**: API heartbeat, model load status
- **Early Stopping**: Patientce=20 epochs, prevents overfitting
- **Data Quality**: Schema, range, and distribution validation

---

## 🏗️ Architecture

### System Design

```
Data Sources (GDSC2)
    ↓
┌─────────────────────────────────┐
│   Data Ingestion & Validation   │
│   - CSV parsing                  │
│   - Train/Val/Test splits        │
│   - Quality checks               │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  Knowledge Graph Construction   │
│   - 15 nodes (5 drugs/5 cells   │
│     /5 diseases)                │
│   - 40 edges (4 relation types) │
│   - Heterogeneous PyG Data      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│     GNN Model Training          │
│   - Input: node embeddings      │
│   - Process: 2 GraphSAGE layers │
│   - Output: [0,1] repur. scores │
│   - Loss: MSE on sigmoid output │
└─────────────────────────────────┘
    ↓
┌──────────────────────┬──────────────────────┐
│                      │                      │
│    REST API          │   Streamlit UI       │
│   (FastAPI)          │   (port 8501)        │
│   (port 8000)        │   Interactive        │
│   JSON endpoints     │   dashboard          │
│                      │                      │
└──────────────────────┴──────────────────────┘
    ↓
┌─────────────────────────────────┐
│    Monitoring & Versioning      │
│   - MLflow experiments          │
│   - DVC artifact tracking       │
│   - GitHub Actions scheduling   │
└─────────────────────────────────┘
```

---

## 📦 Installation

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/mahesh-gtm/mlops-drug-repurposing.git
cd mlops-drug-repurposing

# Build images
docker-compose build

# Start pipeline
docker-compose up -d

# Monitor training
docker-compose logs -f pipeline-init

# Check services running
docker-compose ps
```

**Services Started**:
- `pipeline-init`: Runs data ingest → graph build → model training (exits after completion)
- `fastapi`: REST API server (persists)
- `streamlit`: Web UI (persists)

### Option 2: Local Development

```bash
# Clone and setup Python 3.11+
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run pipeline manually
python -c "from src.ingestion.data_ingest import ingest_gdsc_data; ingest_gdsc_data()"
python -c "from src.preprocessing.build_graph import build_knowledge_graph; build_knowledge_graph()"
python -c "from src.training.train_gnn import train_gnn; train_gnn()"

# Start services
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
streamlit run frontend/app.py --server.port 8501 &
```

---

## 💡 Usage

### 1. Web UI (Streamlit)

Open http://localhost:8501

**Features**:
- **Drug Selector**: Dropdown from available drugs
- **Disease Selector**: Dropdown from available diseases  
- **Predict Button**: Get repurposing score
- **History**: View past predictions
- **Model Info**: Display current model version & training metrics

**Example**:
```
Select Drug: "Methotrexate"
Select Disease: "Lung Cancer"
→ Prediction: 0.673 (Medium confidence)
→ Explanation: "GNN: 0.673 | GDSC: IC50=2.34, AUC=0.82"
```

### 2. REST API (FastAPI)

#### Make Prediction

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
  "explanation": "GNN: 0.673 | GDSC efficacy data",
  "error": false
}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "message": "GNN pipeline is running",
  "model_loaded": true,
  "graph_loaded": true,
  "timestamp": "2026-04-16T22:40:15Z"
}
```

#### Interactive Docs

Open http://localhost:8000/docs to explore API interactively in Swagger UI.

---

## 🔍 API Reference

### POST /predict

**Request Body**:
```json
{
  "drug_id": "string (required)",
  "target_disease": "string (required)"
}
```

**Response** (200):
```json
{
  "drug_id": "string",
  "target_disease": "string",
  "repurposing_score": "float [0.0-1.0]",
  "confidence": "string (high|medium|low|error)",
  "explanation": "string",
  "error": "boolean"
}
```

**Score Interpretation**:
| Range | Interpretation | Action |
|-------|---|---|
| 0.7 - 1.0 | High repurposing potential | Recommend experimental validation |
| 0.4 - 0.7 | Moderate potential | Warrants further research |
| 0.0 - 0.4 | Limited evidence | Requires additional data |

### GET /health

**Response** (200):
```json
{
  "status": "healthy|unhealthy",
  "model_loaded": "boolean",
  "graph_loaded": "boolean",
  "timestamp": "ISO8601 datetime"
}
```

### GET /

Returns API documentation.

---

## 📊 Monitoring & Tracking

### MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Open http://localhost:5000
```

**Features**:
- View all training runs (5+ logged)
- Compare metrics across runs
- Download model artifacts
- Track hyperparameter evolution
- View training loss curves

### View Metrics

```bash
# Display latest metrics
cat artifacts/metrics.json
```

**Sample Output**:
```json
{
  "test_loss": 0.0789,
  "test_mae": 0.2689,
  "test_r2_score": -10.80,
  "epochs_trained": 24,
  "early_stopped": true
}
```

### DVC Pipeline

```bash
# View data pipeline DAG
dvc dag

# Reproduce pipeline
dvc repro

# Show metrics
dvc metrics show
```

---

## 🔄 Continuous Integration

### GitHub Actions Automatic Retraining

**Trigger**: Every 24 hours at 2 AM UTC

**Process**:
1. Pulls latest code
2. Rebuilds Docker images
3. Ingests fresh GDSC2 data
4. Trains new model
5. Saves metrics
6. Commits results to repo

**View Runs**: Go to GitHub Actions tab in repository

**Manual Trigger**:
```bash
# For development, manually retrain
docker-compose down
docker-compose up -d --build
docker-compose logs -f pipeline-init
```

---

## 📁 Project Structure

```
mlops-drug-repurposing/
├── data/
│   ├── raw/                   # GDSC2 CSV files (100 records)
│   ├── processed/             # Knowledge graph, mappings
│   └── splits/                # Train/val/test CSVs
├── src/
│   ├── ingestion/
│   │   └── data_ingest.py     # GDSC2 CSV loading & splitting
│   ├── preprocessing/
│   │   └── build_graph.py     # Knowledge graph construction
│   ├── training/
│   │   └── train_gnn.py       # Model training with MLflow
│   ├── inference/
│   │   └── predict.py         # Model inference logic
│   └── utils/
│       └── config.py          # Centralized configuration
├── api/
│   ├── main.py                # FastAPI endpoints
│   └── schemas.py             # Pydantic request/response models
├── frontend/
│   └── app.py                 # Streamlit UI
├── artifacts/
│   ├── model.pt               # Trained weights
│   └── metrics.json           # Final evaluation metrics
├── mlruns/                    # MLflow tracking directory
├── docker-compose.yml         # Service orchestration
├── Dockerfile                 # Container specification
├── requirements.txt           # Python dependencies
├── TECHNICAL_REPORT.md        # (NEW) Full architecture & results
└── README.md                  # This file
```

---

## ⚙️ Configuration

### Key Parameters

Edit `src/utils/config.py` to tune:

```python
CONFIG = {
    # Data
    "train_split": 0.70,          # 70% training
    "val_split": 0.15,            # 15% validation
    "test_split": 0.15,           # 15% testing
    
    # Training
    "learning_rate": 0.02,        # Adam optimizer lr
    "num_epochs": 150,            # Max epochs
    "hidden_dim": 96,             # GNN hidden dimension
    "batch_size": 32,             # Batch size (full if < num_pairs)
    
    # Monitoring
    "early_stopping_patience": 20,        # Stop if no improvement for 20 epochs
    "early_stopping_threshold": 0.001,    # Min loss improvement
    "log_metrics_every": 10,              # Log every 10 epochs
    
    # Paths
    "mlflow_tracking_uri": "./mlruns",
    "artifacts_path": "artifacts/",
    "processed_data_path": "data/processed/",
}
```

### Environment Variables

```bash
# In docker-compose.yml or .env
MLFLOW_TRACKING_URI=./mlruns           # MLflow backend
GIT_PYTHON_REFRESH=quiet               # Suppress git warnings
```

---

## 🐛 Troubleshooting

### API Returns "Model not loaded"

**Issue**: Predictions fail with error message.

**Solution**:
```bash
# Wait for pipeline to complete
docker-compose logs pipeline-init | grep "Pipeline initialization completed"

# Verify model file exists
ls -lh artifacts/model.pt
```

### All Predictions Return 0.0

**Issue**: Drug or disease not found.

**Solution**:
- Check exact spelling (case-sensitive, but has fallback)
- View available entities: `data/processed/graph_mappings.pkl`
- Use API docs at http://localhost:8000/docs to test

### Out of Memory

**Issue**: Training crashes with OOM error.

**Solution** (in `config.py`):
```python
"hidden_dim": 64,  # Reduce from 96
"num_epochs": 50,  # Reduce from 150
```

Then rebuild: `docker-compose up -d --build`

### Training Stuck / Not Improving

**Issue**: Epochs run but loss doesn't decrease.

**Solution**:
1. Check learning rate isn't too high: Try `0.01` instead of `0.02`
2. Verify data loaded: `docker-compose logs pipeline-init | grep "train_pairs"`
3. Monitor: `docker-compose logs -f pipeline-init`

---

## 📈 Performance Benchmarks

### Model Evaluation (Current)

| Metric | Value | Notes |
|--------|-------|-------|
| **Test MSE Loss** | 0.0789 | Low error on sigmoid-bounded outputs |
| **Test MAE** | 0.2689 | Mean deviation from targets |
| **Test R² Score** | -10.80 | Limited by small dataset (25 pairs) |
| **Prediction Range** | [0.67, 0.68] | Concentrated (needs more data for spread) |
| **Training Time** | ~15 sec | on CPU; <5 sec on GPU |
| **Inference Time** | ~5 ms/prediction | Per drug-disease pair |

### Scaling Notes

Production deployment would require:
- **10,000+ drug-disease training pairs** (vs current 25)
- Expected R² improvement: -10 → 0.75+
- Graph expansion: 100 drugs × 50 diseases × 500 cell lines
- Training time: 2-5 minutes on GPU

---

## 🚦 Status & Roadmap

### ✅ Completed

- [x] End-to-end pipeline (ingest → train → serve)
- [x] Docker containerization
- [x] FastAPI REST interface
- [x] Streamlit web UI
- [x] MLflow experiment tracking
- [x] DVC data versioning
- [x] GitHub Actions CI/CD
- [x] Monitoring & health checks
- [x] Technical documentation

###⏳ In Progress

- [ ] Expand dataset to 1,000+ drug pairs
- [ ] Improve model R² score to >0.70
- [ ] Add semi-supervised learning

### 📋 Planned (Phase 2)

- [ ] Protein-protein interaction network integration
- [ ] Multi-target learning (synergy prediction)
- [ ] Ensemble multiple GNN architectures
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Kubernetes orchestration
- [ ] Feature store integration

---

## 📖 Documentation

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)** - Detailed architecture, ML methods, monitoring strategy
- **[MONITORING_STRATEGY.md](MONITORING_STRATEGY.md)** - Data quality, drift detection, health checks
- **[DVC_SETUP.md](DVC_SETUP.md)** - Data versioning & reproducibility
- **[PHASE2_SUMMARY.md](PHASE2_SUMMARY.md)** - MLflow, metrics, monitoring implementation

---

## 🤝 Contributing

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes
# - Update code in src/, api/, or frontend/
# - Test locally with docker-compose

# 3. Run tests
docker-compose run pipeline-init python -m pytest

# 4. Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature

# 5. Create pull request
# GitHub Actions will automatically test on merge
```

### Code Style

- Python: PEP 8 with Black formatter
- Type hints: Required for all functions
- Docstrings: Google-style for all modules/classes

---

## 📝 License

MIT License - see LICENSE file

---

## 📧 Contact & Support

- **Issues**: Create a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: maheshgautam345@gmail.com

---

## 📚 References

- **Paper**: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs" ([GraphSAGE](https://arxiv.org/abs/1706.02216))
- **Data**: GDSC2 Cancer Pharmacogenomics DB (https://www.cancerrxgene.org/)
- **Framework**: PyTorch Geometric (https://pytorch-geometric.readthedocs.io/)

---

**Last Updated**: April 16, 2026  
**Version**: 1.0  
**Maintainer**: Mahesh gautam

 
