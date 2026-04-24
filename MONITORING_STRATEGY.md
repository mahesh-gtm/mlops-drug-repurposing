# MLOps Monitoring Strategy - Drug Repurposing GNN Pipeline

**Version**: 1.0  
**Last Updated**: April 2026  
**Status**: In Production

---

## 1. Overview

This document outlines the comprehensive monitoring and observability strategy for the Drug Repurposing GNN MLOps pipeline. The strategy covers model performance monitoring, data quality checks, system health, and continuous model evaluation.

### Key Objectives
- **Real-time Performance Tracking**: Monitor model metrics during training and inference
- **Data Quality Assurance**: Detect anomalies in input data and drift
- **System Reliability**: Track pipeline execution and system health
- **Continuous Improvement**: Identify retraining triggers and model degradation
- **Reproducibility**: Maintain complete audit trail of all runs and experiments

---

## 2. Monitoring Architecture

### 2.1 Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Stack                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │   MLflow Tracking    │    │   Metrics/Artifacts      │  │
│  │  (Experiments, Runs) │    │   (Metrics.json, DVC)    │  │
│  └──────────────────────┘    └──────────────────────────┘  │
│           ▲                            ▲                    │
│           │                            │                    │
│  ┌────────┴────────┬──────────────────┴─────────┐          │
│  │                 │                            │           │
│  │    Training    │      Inference             │           │
│  │    Pipeline    │      Endpoints             │           │
│  │                 │                            │           │
│  └─────────────────┴────────────────────────────┘          │
│           ▲                    ▲                             │
│           │                    │                             │
│  ┌────────┴────────┐   ┌──────┴──────────┐                 │
│  │  Data Ingestion │   │  API (FastAPI)  │                 │
│  │  & Validation   │   │  & Streamlit    │                 │
│  └─────────────────┘   └─────────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Tech Stack
- **MLflow**: Experiment tracking, model registry, metrics storage
- **DVC**: Data versioning and artifact management
- **GitHub Actions**: Workflow orchestration and scheduling
- **Evidently**: Data quality and model drift detection (planned)
- **JSON metrics**: Simple, version-controllable metrics storage

---

## 3. Training Monitoring

### 3.1 Metrics Tracked

During model training, the following metrics are automatically collected:

**Primary Metrics:**
| Metric | Definition | Target | Alert Threshold |
|--------|-----------|--------|-----------------|
| `train_loss` | MSE on training data | ↓ Decreasing | > 0.5 |
| `val_loss` | MSE on validation data | ↓ Decreasing | > 0.3 |
| `test_loss` | MSE on held-out test set | ↓ Minimize | > 0.4 |
| `test_mae` | Mean Absolute Error | ↓ Minimize | > 0.5 |
| `test_r2_score` | R² coefficient | ↑ Maximize | < 0.5 |

**Secondary Metrics:**
- `epochs_trained`: Total epochs completed
- `early_stopped`: Whether early stopping was triggered
- `num_parameters`: Total trainable model parameters
- `best_val_loss`: Best validation loss achieved

### 3.2 MLflow Integration

#### Run Parameters Logged
```python
Parameters:
  - num_epochs: 50
  - learning_rate: 0.01
  - hidden_dim: 64
  - batch_size: 32
  - train_split: 0.70
  - val_split: 0.15
  - test_split: 0.15
```

#### Metrics Logged per Epoch
```python
Every 10 epochs:
  - train_loss
  - val_loss

End of training:
  - test_loss
  - test_mae
  - test_r2_score
  - final_train_loss
  - final_val_loss
  - best_val_loss
```

#### Artifacts Stored
- `model.pt`: Trained model weights
- `metrics.json`: Complete evaluation results
- Model object (via `mlflow.pytorch.log_model`)

### 3.3 Early Stopping

**Strategy**: Validation-based early stopping
- **Patience**: 20 epochs without improvement
- **Threshold**: 0.001 (minimum improvement required)
- **Benefit**: Prevents overfitting and saves training time

### 3.4 How to View Training Results

```bash
# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Then navigate to: http://localhost:5000
# View:
#  - Experiment: "drug-repurposing-gnn"
#  - Compare runs side-by-side
#  - Download metrics for analysis
```

---

## 4. Inference Monitoring

### 4.1 Endpoint Metrics

**FastAPI Prediction Endpoint**: `POST /predict`

Tracked metrics:
- Response time (latency)
- Prediction score distribution
- Error rate
- Request frequency

### 4.2 Health Checks

**Endpoint**: `GET /health`

Response:
```json
{
  "status": "healthy",
  "message": "GNN pipeline is running",
  "model_loaded": true,
  "graph_loaded": true,
  "timestamp": "2026-04-16T10:30:00Z"
}
```

### 4.3 Prediction Logging

Each prediction is logged with:
- Input: drug_id, target_disease
- Output: repurposing_score, confidence
- Timestamp
- Model version
- Inference time

---

## 5. Data Quality Monitoring

### 5.1 Data Validation Checklist

**Ingestion Stage**:
- ✅ File existence and format validation
- ✅ Missing value detection
- ✅ Schema validation (expected columns)
- ✅ Numeric range validation (IC50, AUC bounds)

**Implemented in**: `src/ingestion/data_ingest.py`

### 5.2 Data Distribution Monitoring

Track: 
- Mean/std of IC50 values per time window
- Number of unique drugs/cell lines
- New vs. known entities ratio
- Data freshness (age of newest records)

### 5.3 Statistical Tests

Planned integrations (Evidently):
- **Drift Detection**: Kolmogorov-Smirnov test for distribution shifts
- **Outlier Detection**: IQR-based anomaly detection
- **Completeness**: Monitor null values over time

---

## 6. Model Performance Monitoring

### 6.1 Performance Baseline

**Baseline** established on first training:
- Test MSE: From first production run
- Test MAE: From first production run
- Test R²: From first production run

### 6.2 Drift Detection Triggers

Retrain if ANY trigger is met:

**Performance Degradation**:
- If current test MSE > baseline + 20%
- If test R² < 0.5 for 3 consecutive predictions

**Data Drift**:
- If new data distribution differs significantly
- If new entity types detected (unknown drugs/cells)

**Schedule**:
- Automatic daily check via GitHub Actions (03:00 UTC)
- Manual trigger available anytime
- Production deployment after validation

### 6.3 Model Version Management

**Versioning Strategy**:
```
artifacts/
├── model.pt                 # Current production model
├── model_v1_2026-01-15.pt   # Archived version
├── model_v2_2026-02-20.pt   # Archived version
├── metrics.json             # Current metrics
└── metrics_history.jsonl    # Historical metrics
```

**Rollback Procedure**:
1. Identify degraded model version
2. Locate best previous artifact in MLflow
3. Update `model_path` in config
4. Redeploy container with pinned version
5. Verify with health check

---

## 7. Pipeline Execution Monitoring

### 7.1 Scheduled Retraining

**Workflow**: `.github/workflows/retrain.yml`

**Schedule**: Daily at 03:00 UTC
- [x] Ingest latest data
- [x] Build updated knowledge graph
- [x] Train new model
- [x] Evaluate on test set
- [x] Compare with baseline
- [x] Log metrics to MLflow

**Failure Scenarios Handled**:
1. Data ingestion failure → Alert + skip training
2. Graph building failure → Alert + use previous graph
3. Training failure → Alert + keep current model
4. All failures logged to GitHub Actions UI

### 7.2 Manual Triggers

```bash
# Trigger retraining from GitHub UI or CLI:
gh workflow run retrain.yml

# Benefits:
# - Test new hyperparameters
# - Quick response to urgent issues
# - On-demand model updates
```

---

## 8. Alerting Strategy

### 8.1 Alert Conditions

| Alert Level | Condition | Action |
|-------------|-----------|--------|
| 🔴 Critical | Model inference fails | Immediate page-on-call |
| 🟠 High | Test loss > baseline + 50% | Team notification |
| 🟡 Medium | Data drift detected | Log & schedule review |
| 🔵 Low | Training took > 2x baseline | Log & monitor |

### 8.2 Notification Channels

- GitHub Actions: Native email notifications
- Metrics Dashboard: MLflow UI for detailed analysis
- Logs: Centralized in `artifacts/` directory

---

## 9. Reproducibility & Audit Trail

### 9.1 Complete Tracking

Every training run captures:
```json
{
  "run_id": "abc123def456",
  "timestamp": "2026-04-16T03:00:00Z",
  "git_commit": "a1b2c3d4e5f6",
  "data_version": "train_split_v2",
  "hyperparameters": {...},
  "metrics": {...},
  "model_artifact": "path/to/model.pt",
  "git_branch": "main"
}
```

### 9.2 DVC Integration

**Planned enhancements**:
- [ ] Track data versions in DVC
- [ ] Remote storage configuration
- [ ] Model registry with DVC pipelines
- [ ] Automatic versioning triggers

---

## 10. Dashboard & Visualization

### 10.1 MLflow Dashboard

Navigate to `http://localhost:5000` after running:
```bash
mlflow ui --backend-store-uri ./mlruns
```

**Available views**:
- Experiment overview
- Metrics vs. Epoch graphs
- Parameter comparison across runs
- Artifact browser

### 10.2 Streamlit Frontend

Access at `http://localhost:8501`

**Monitoring Features**:
- Real-time prediction scores
- Knowledge graph visualization
- Model confidence levels
- Latest metrics display

---

## 11. Incident Response

### 11.1 Model Degradation

**Detection**: Test loss increases by >20%

**Response**:
1. Check data ingestion for anomalies
2. Review recent data distributions
3. Retrain with current hyperparameters
4. If no improvement, investigate known issues
5. Rollback to previous model if necessary

### 11.2 Data Quality Issues

**Detection**: Evidently drift alerts OR manual review

**Response**:
1. Validate data source integrity
2. Check for upstream pipeline failures
3. Quarantine suspicious records
4. Resume pipeline with clean data

### 11.3 System Failures

**Detection**: Health check failures OR GitHub Actions alert

**Response**:
1. Check compute resources (disk, memory, CPU)
2. Verify dependencies in requirements.txt
3. Review recent code changes
4. Redeploy from last known-good state

---

## 12. Implementation Roadmap

### Current (✅ Implemented)
- [x] MLflow experiment tracking
- [x] Train/val/test metrics logging
- [x] Early stopping with patience
- [x] Metrics JSON storage
- [x] GitHub Actions scheduled runs
- [x] Health check endpoint
- [x] Model artifact versioning

 

---

## 13. Key Contacts & Documentation

- **Repository**: https://github.com/mahesh-gtm/mlops-drug-repurposing
- **MLflow Server**: `./mlruns` (local)
- **Runbook**: See `README.md` for debugging
- **Issues**: GitHub Issues for tracking bugs/feature requests

---

## 14. Appendix: Metric Interpretation Guide

### R² Score
- **1.0** = Perfect predictions
- **0.5** = Model explains 50% of variance
- **0.0** = Model performs like simple mean
- **< 0.0** = Model is worse than baseline

### MAE vs MSE
- **MAE**: Average absolute error (more interpretable)
- **MSE**: Penalizes large errors more (sensitive to outliers)
- Use both for complete picture

### Early Stopping
- **Triggers when** validation loss plateaus
- **Purpose** prevents overfitting to training data
- **Trade-off** trains fewer epochs but better generalization

---

**Last Updated**: April 16, 2026  
**Next Review**: May 16, 2026
