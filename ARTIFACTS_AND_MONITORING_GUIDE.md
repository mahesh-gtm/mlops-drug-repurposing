# Artifacts & Monitoring Guide

## 1. What is the Artifacts Folder?

The `artifacts/` folder stores all **trained model outputs and metrics** from each training run.

### Files Stored:
```
artifacts/
├── model.pt              # Trained neural network weights (328 KB)
│   └── Contains: GNN model with all learned parameters
│       - Usable for inference without retraining
│       - Loaded by FastAPI every time predictions are made
│
└── metrics.json          # Final evaluation metrics (369 bytes)
    └── Contains: Performance scores from test set
        - test_loss: 0.0789 (MSE error)
        - test_mae: 0.2689 (absolute error)
        - test_r2_score: -10.80 (variance explained)
        - epochs_trained: 25
        - And more...
```

### Why Separate from MLflow?
- **artifacts/** = Production model files (used by API)
- **mlruns/** = Experiment history (for analysis/comparison)

---

## 2. How to Check Artifacts

### Option 1: View Files Directly
```bash
# Check what's in artifacts
ls -lah artifacts/

# See file sizes
du -sh artifacts/*

# View metrics
cat artifacts/metrics.json
```

**Output**:
```
artifacts/
-rw-rw-rw- 1 root root  369 Apr 16 22:41 metrics.json
-rw-rw-rw- 1 root root  326K Apr 16 22:41 model.pt
```

### Option 2: Check Metrics Programmatically
```bash
# Pretty-print metrics
cd artifacts && python3 -m json.tool metrics.json

# Extract specific metric
cat artifacts/metrics.json | grep test_mae
```

**Output**:
```json
{
  "final_train_loss": 0.0154,
  "final_val_loss": 0.0510,
  "test_loss": 0.0789,
  "test_mae": 0.2689,        ← Mean Absolute Error
  "test_r2_score": -10.80,   ← Variance Explained
  "epochs_trained": 25,
  "early_stopped": true
}
```

### Option 3: Monitor in Real-time During Training
```bash
# Watch metrics as training runs
watch -n 1 'cat artifacts/metrics.json'

# Or check model file size growth
watch -n 1 'ls -lh artifacts/model.pt'
```

### Option 4: Load Model in Python
```python
import torch
import os

# Check model file size
size_mb = os.path.getsize('artifacts/model.pt') / 1024 / 1024
print(f"Model size: {size_mb:.2f} MB")

# Load model to inspect
model = torch.load('artifacts/model.pt', weights_only=False)
print(f"Model type: {type(model)}")
print(f"Model keys: {model.keys() if hasattr(model, 'keys') else 'Not a dict'}")
```

---

## 3. Why Monitoring URL (http://localhost:5000) Not Working

### ❌ Problem
MLflow stores metrics, but the **MLflow UI server is NOT running**.

### Current Status
```bash
# These ARE listening:
✅ Port 8000   (FastAPI API)
✅ Port 8501   (Streamlit UI)

# This is NOT listening:
❌ Port 5000   (MLflow UI) ← NOT RUNNING
```

### Root Cause
MLflow is only used for **logging** data (during training), not for **serving** the UI.
- Data saved to: `mlruns/` directory
- UI server: Needs to be started manually

---

## Solution 1: Start MLflow UI (Recommended)

### Quick Start (Single Command)
```bash
# From project root directory
mlflow ui --backend-store-uri ./mlruns

# Then open: http://localhost:5000
```

**What it does**:
1. Starts MLflow server on port 5000
2. Reads from `mlruns/` directory
3. Shows:
   - All past training runs
   - Metrics graphs
   - Parameter comparisons
   - Model artifacts download

### Keep it Running
```bash
# Run in background
nohup mlflow ui --backend-store-uri ./mlruns > mlflow.log 2>&1 &

# Or in a separate terminal (easier to monitor)
# Terminal 1: docker-compose logs -f
# Terminal 2: mlflow ui --backend-store-uri ./mlruns
```

### View UI
Open browser: **http://localhost:5000**

**You'll see**:
- Experiment: "drug-repurposing-gnn"
- Runs: 6+ training runs with date/time
- Metrics: train_loss, val_loss, test_loss graphs
- Parameters: learning_rate, hidden_dim, etc.
- Artifacts: Download model.pt, metrics.json

---

## Solution 2: Add MLflow to Docker Compose (Advanced)

### Create MLflow Container
Edit `docker-compose.yml`, add:

```yaml
mlflow:
  image: python:3.11-slim
  working_dir: /app
  command: bash -c "pip install mlflow && mlflow ui --backend-store-uri file:///app/mlruns --host 0.0.0.0"
  ports:
    - "5000:5000"
  volumes:
    - ./mlruns:/app/mlruns
  depends_on:
    - pipeline-init
```

Then:
```bash
docker-compose up -d mlflow
# Now MLflow auto-starts with other services
```

**Advantage**: MLflow starts automatically with `docker-compose up`

---

## Verification Checklist

### ✅ Artifacts Working
```bash
[ ] artifacts/model.pt exists (200+ KB)
[ ] artifacts/metrics.json exists
[ ] Metrics JSON has test_loss, test_mae, test_r2_score
[ ] FastAPI can load model (test: curl http://localhost:8000/health)
```

### ✅ Monitoring Working
```bash
[ ] Port 5000 shows MLflow UI running
[ ] Can see "drug-repurposing-gnn" experiment
[ ] Can see 6+ training runs listed
[ ] Metrics graphs display
[ ] Can compare runs side-by-side
```

---

## Quick Reference

### Check Everything
```bash
# 1. Artifacts present?
ls -lh artifacts/

# 2. Metrics cached?
cat artifacts/metrics.json | jq .test_r2_score

# 3. API working?
curl http://localhost:8000/health

# 4. Start MLflow monitoring
mlflow ui --backend-store-uri ./mlruns

# 5. Open browser
open http://localhost:5000
```

### Troubleshooting

**Q: Model file is very small (< 10KB)**
```bash
❌ Model likely corrupted or incomplete
✅ Check: docker-compose logs pipeline-init | grep "model saved"
```

**Q: metrics.json very old (not recent)**
```bash
❌ Model not retraining
✅ Check: docker-compose logs pipeline-init | tail -50
```

**Q: MLflow says "No runs found"**
```bash
❌ mlruns/ directory empty or wrong path
✅ Check: ls -la mlruns/
✅ Solution: Re-run training: docker-compose up -d --build
```

**Q: Port 5000 already in use**
```bash
# Find what's using it
lsof -i :5000

# Use different port
mlflow ui --backend-store-uri ./mlruns --port 6000
# Then open: http://localhost:6000
```

---

## Summary

| Component | What It Does | How to Access |
|-----------|---|---|
| **artifacts/model.pt** | Trained weights used by API | Load directly or via API |
| **artifacts/metrics.json** | Final test metrics | View as JSON file |
| **mlruns/** | All experiment history | MLflow UI (Port 5000) |
| **Port 5000** | MLflow web dashboard | `mlflow ui --backend-store-uri ./mlruns` |
| **Port 8000** | API predictions | `curl http://localhost:8000/predict` |
| **Port 8501** | Streamlit UI | Open browser http://localhost:8501 |

---

## Next Steps

1. **View Artifacts**: `cat artifacts/metrics.json`
2. **Start Monitoring**: `mlflow ui --backend-store-uri ./mlruns`
3. **Open Monitoring**: http://localhost:5000
4. **View All Runs**: Click "drug-repurposing-gnn" experiment
5. **Get Predictions**: http://localhost:8501 (Streamlit UI)

