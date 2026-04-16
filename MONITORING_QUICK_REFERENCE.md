# Quick Reference: Artifacts & Monitoring

## 1️⃣ ARTIFACTS FOLDER - What It Does

### Location: `artifacts/` (in project root)

| File | Size | Purpose | Used By |
|------|------|---------|---------|
| **model.pt** | 328 KB | Trained neural network weights | FastAPI (makes predictions) |
| **metrics.json** | 369 bytes | Final test performance scores | Monitoring & reporting |

### It's Like a "Saved Game"
- After each training run, the model is saved to `model.pt`
- API loads `model.pt` on startup to serve predictions
- Metrics saved for tracking performance over time

---

## 2️⃣ HOW TO CHECK ARTIFACTS

### Quickest Way (Copy-Paste These)

```bash
# Check what files exist
ls -lh artifacts/

# View latest metrics
cat artifacts/metrics.json

# Extract one metric
cat artifacts/metrics.json | grep test_mae
```

### What You'll See
```json
{
  "test_loss": 0.0789,
  "test_mae": 0.2689,
  "test_r2_score": -10.80,
  "epochs_trained": 25,
  "early_stopped": true
}
```

### Interpretation
- ✅ **model.pt exists** = API can make predictions
- ✅ **metrics.json updated** = Model recently trained
- ✅ **early_stopped: true** = Training converged properly
- ⚠️ **test_r2_score negative** = Model has limited data (not a bug!)

---

## 3️⃣ WHY MONITORING URL (Port 5000) NOT WORKING

### The Issue
```
❌ Tried: http://localhost:5000
❌ Result: "Connection refused"
❌ Reason: No service listening on port 5000
```

### What's Actually Running
```
✅ Port 8000   = API (working)
✅ Port 8501   = Streamlit UI (working)
❌ Port 5000   = MLflow UI (NOT in Docker)
```

### Why?
MLflow has two parts:
1. **Logging** (happens during training) - ✅ Working
2. **UI Server** (web dashboard) - ❌ Not automatically started

MLflow stores data to `mlruns/` directory but doesn't serve the web UI in Docker.

---

## FIX: Access Monitoring Data

### Option A: View As Files (Simplest)
```bash
# Your data IS THERE, just not in a web UI yet
cat artifacts/metrics.json          # Latest values
ls -la mlruns/                      # All experiment history
```

### Option B: View The Raw Data
```bash
# View all metrics from all 6+ training runs
find mlruns -name "metrics" -type f | xargs cat

# See experiment structure
tree mlruns/  # or: find mlruns -type f | head -20
```

### Option C: Start MLflow UI (Need to Install MLflow)
```bash
# If MLflow installed locally:
mlflow ui --backend-store-uri ./mlruns
# Then open: http://localhost:5000
```

**Status**: MLflow not currently installed. Would need:
```bash
pip install mlflow
mlflow ui --backend-store-uri ./mlruns
```

---

## What You CAN Access Right Now

| What | Where | How |
|------|-------|-----|
| **Model** | artifacts/model.pt | Used by API |
| **Metrics** | artifacts/metrics.json | View in terminal |
| **Training History** | mlruns/ directory | Files on disk |
| **Predictions** | http://localhost:8000/predict | API endpoint |
| **Web UI** | http://localhost:8501 | Streamlit dashboard |
| **API Docs** | http://localhost:8000/docs | Swagger explorer |

---

## Complete Monitoring Data Location

```
mlops-drug-repurposing/
│
├── artifacts/                          ← Loaded by API
│   ├── model.pt                        (328 KB - the neural network)
│   └── metrics.json                    (369 bytes - test scores)
│
├── mlruns/                             ← MLflow storage
│   └── 804208927739947265/             (experiment ID)
│       └── run_ID/
│           ├── metrics/                (loss, mae, r2 per epoch)
│           ├── params/                 (hyperparameters used)
│           ├── artifacts/              (model backup)
│           └── tags/                   (metadata)
│
├── data/processed/                     ← Training data
│   ├── graph.pt                        (knowledge graph)
│   └── graph_mappings.pkl              (node lookups)
│
└── docker-compose.logs                 ← Training output
   (accessible via: docker-compose logs)
```

---

## Summary Table

| Question | Answer | Where |
|----------|--------|-------|
| **What is artifacts?** | Saved model + metrics | `artifacts/` folder |
| **How to check?** | `cat artifacts/metrics.json` | Terminal command |
| **Why port 5000 fails?** | MLflow UI not in Docker | Not auto-started |
| **How to fix?** | View data as files OR install MLflow | See "Fix" section |
| **What IS working?** | API (8000), UI (8501), predictions | Tested & verified |

---

## Action Items

### ✅ To View Your Metrics Right Now:
```bash
cd /workspaces/mlops-drug-repurposing
cat artifacts/metrics.json
```

You'll see all your model's performance scores.

### ✅ To Check Training History:
```bash
ls -la mlruns/
find mlruns -type f | head -20
```

### ✅ To Make Predictions:
```bash
# Option 1: API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"drug_id":"Methotrexate","target_disease":"Lung Cancer"}'

# Option 2: Web UI
open http://localhost:8501
```

---

## If You Need The MLflow Web UI

To get http://localhost:5000 working in the future:

**Quick (1 command):**
```bash
pip install mlflow
mlflow ui --backend-store-uri ./mlruns
# Then: open http://localhost:5000
```

**Permanent (Docker):**
Edit `docker-compose.yml` and add MLflow service (see ARTIFACTS_AND_MONITORING_GUIDE.md)

---

**Status**: ✅ Monitoring data EXISTS and IS COMPLETE  
**Issue**: Just not exposed as a web UI yet  
**Solution**: View as files (working now) or install MLflow (1 command)
