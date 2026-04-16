# Phase 2 Quick Reference Guide

**Quick access commands for MLflow, DVC, and monitoring features**

---

## 🚀 Quick Start After Phase 2

### 1️⃣ Run Full Pipeline with Monitoring

```bash
# Docker (Recommended)
docker-compose up --build

# Or locally (needs Python/dependencies)
dvc repro
```

Then:
```bash
# View MLflow dashboard
mlflow ui --backend-store-uri ./mlruns
# Open: http://localhost:5000
```

### 2️⃣ Access the Running Services

```
FastAPI:    http://localhost:8000
  - API Docs: http://localhost:8000/docs
  - Health:   http://localhost:8000/health
  - Predict:  POST http://localhost:8000/predict

Streamlit:  http://localhost:8501
  - Interactive UI for predictions

MLflow:     http://localhost:5000
  - Experiment tracking
  - Metrics comparison
  - Model artifacts
```

---

## 📊 Viewing Training Results

### MLflow Dashboard

```bash
mlflow ui --backend-store-uri ./mlruns
```

**View**:
- ✓ Experiment name: "drug-repurposing-gnn"
- ✓ Metrics graphs (train_loss, val_loss, test_r2_score)
- ✓ Parameter comparison
- ✓ Model artifacts
- ✓ Run timestamps

### Metrics File

```bash
# View last training results
cat artifacts/metrics.json
```

**Output**:
```json
{
  "final_train_loss": 0.214,
  "final_val_loss": 0.245,
  "test_loss": 0.259,
  "test_mae": 0.452,
  "test_r2_score": 0.623,
  "epochs_trained": 35,
  "early_stopped": true,
  "num_parameters": 32896
}
```

---

## 🔄 Running Pipeline Stages

### Run Full Pipeline

```bash
dvc repro
# Runs: ingest → build_graph → train
# Only reruns stages with changed dependencies
```

### Run Specific Stage

```bash
# Only retrain model (skip data processing)
dvc repro -s train

# Force rerun all stages
dvc repro --force

# Dry run (show what would run)
dvc repro --dry
```

### Check Pipeline Status

```bash
dvc status
# Output:
#   data/raw/gdsc2_original.csv
#     modified: true
#       ...

dvc dag
# Shows pipeline dependency graph
```

---

## 🎯 Monitoring & Debugging

### Check Training Logs

```bash
# Last 50 lines of output
tail -50 docker-compose logs fastapi

# Follow logs in real-time
docker-compose logs -f pipeline-init
```

### View Data Splits

```bash
ls -lh data/splits/
# train.csv    (70%)
# val.csv      (15%)
# test.csv     (15%)

# Check split distribution
wc -l data/splits/*.csv
```

### Check Graph Statistics

```bash
cat data/processed/graph_stats.json
# Shows: nodes, edges, graph density, connectivity
```

---

## 🔐 MLflow Integration

### View Specific Experiment

```bash
# List all experiments
mlflow experiments list

# View specific run
mlflow runs describe <run_id>

# Compare two runs
mlflow runs compare <run_id_1> <run_id_2>
```

### Download Metrics Programmatically

```python
import mlflow

client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("drug-repurposing-gnn")
runs = client.search_runs(experiment.experiment_id)

for run in runs:
    print(f"Run: {run.info.run_id}")
    print(f"Metrics: {run.data.metrics}")
    print(f"Params: {run.data.params}")
```

---

## 📦 DVC Commands

### Track New Data

```bash
# Add new dataset
dvc add data/raw/new_data.csv

# Commit .dvc file to Git
git add data/raw/new_data.csv.dvc
git commit -m "Add new GDSC batch"
```

### Status & Tracking

```bash
# Show all tracked files
dvc list .

# Show what changed
dvc diff

# Compare with specific commit
dvc diff abc123def
```

### Push/Pull Artifacts

```bash
# Push to remote storage
dvc push

# Pull from remote storage
dvc pull

# Restore specific file
dvc checkout data/processed/graph.pt
```

---

## 🧪 Testing the API

### Health Check

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "message": "GNN pipeline is running"
}
```

### Make Prediction

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
  "repurposing_score": 0.65,
  "confidence": "medium",
  "explanation": "..."
}
```

### Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "drug_id": "Paclitaxel",
        "target_disease": "Breast Cancer"
    }
)
print(response.json())
```

---

## 🐛 Troubleshooting

### Issue: "Model not loaded"

```bash
dvc pull                        # Restore artifacts
dvc repro                       # Retrain if needed
docker-compose restart fastapi  # Restart service
```

### Issue: "Pipeline is out of date"

```bash
dvc status                      # Check status
dvc repro --force              # Force retrain
```

### Issue: "No MLflow runs visible"

```bash
# Ensure tracking URI is set
export MLFLOW_TRACKING_URI=./mlruns

# Start UI pointing to correct location
mlflow ui --backend-store-uri ./mlruns

# Verify runs exist
ls -la mlruns/
```

### Issue: Container won't start

```bash
# Remove dangling containers
docker-compose down -v

# Rebuild fresh
docker-compose up --build

# Check logs
docker-compose logs pipeline-init
```

---

## 📚 Reference Documentation

**Just Created (Phase 2)**:
- [Monitoring Strategy](./MONITORING_STRATEGY.md) - Full monitoring setup
- [DVC Setup Guide](./DVC_SETUP.md) - Data versioning workflow
- [Phase 2 Summary](./PHASE2_SUMMARY.md) - What was implemented

**Existing**:
- [README.md](./README.md) - Basic overview
- [GitHub Actions](../.github/workflows/retrain.yml) - Auto retraining

---

## 🎯 Next Steps

### Before Submission:
- [ ] Create technical report PDF
- [ ] Expand README with setup instructions
- [ ] Deploy Streamlit frontend publicly
- [ ] Add screenshots to technical report

### Optional:
- [ ] Add unit tests
- [ ] Set up S3 remote for DVC
- [ ] Create Grafana dashboard
- [ ] Implement data drift detection

---

## 🔑 Key Metrics Interpretation

| Metric | Good Value | Poor Value | Action |
|--------|-----------|-----------|--------|
| test_r2_score | > 0.6 | < 0.3 | Retrain |
| test_mae | < 0.5 | > 1.0 | Check data |
| test_loss | Decreasing | Increasing | Early stop |
| early_stopped | True | False | May overfit |
| inference_latency | < 100ms | > 500ms | Optimize |

---

## ✅ Verification Checklist

After running Phase 2:

- [ ] `artifacts/metrics.json` exists and has values
- [ ] `data/splits/` has train.csv, val.csv, test.csv
- [ ] `data/processed/graph_stats.json` shows graph info
- [ ] `mlruns/` directory has experiment data
- [ ] `dvc status` shows no conflicts
- [ ] MLflow UI loads at http://localhost:5000
- [ ] FastAPI docs at http://localhost:8000/docs
- [ ] Streamlit UI at http://localhost:8501

---

**Status**: ✅ All Phase 2 features ready to use  
**Last Updated**: April 16, 2026
