# DVC Setup & Data Versioning Guide

**Version**: 1.0  
**Last Updated**: April 2026

---

## 1. Overview

This project uses **DVC (Data Version Control)** to manage:
- **Data artifacts** (raw data, processed graphs)
- **Model artifacts** (trained weights, metrics)
- **Reproducibility** across runs and team members
- **Pipeline orchestration** (ingest → preprocess → train)

---

## 2. DVC Architecture

### 2.1 Current Setup

```
Project root/
├── .dvc/
│   ├── config                    # DVC configuration
│   └── .gitignore               # DVC cache ignores
├── .dvcignore                   # Files to exclude from tracking
├── dvc.yaml                     # Pipeline definition
├── dvc.lock                     # Locked dependency versions (generated)
└── data/
    ├── raw/                    # Raw ingested data (DVC tracked)
    ├── processed/              # Processed graphs (DVC tracked)
    └── splits/                 # Train/val/test splits (DVC tracked)
```

### 2.2 Remote Storage Options

Currently configured: **Local storage** (`.dvc/cache/storage`)

For production, consider:
- **AWS S3**: `s3://bucket-name/dvc-storage`
- **Google Cloud Storage**: `gs://bucket-name/dvc-storage`
- **Azure Blob Storage**: `azure://container/dvc-storage`
- **GitHub**: Native DVC integration

---

## 3. Pipeline Definition (dvc.yaml)

The pipeline has three stages:

### Stage 1: Data Ingestion (`ingest`)
```yaml
cmd: python src/ingestion/data_ingest.py
deps:
  - src/ingestion/data_ingest.py
  - src/utils/config.py
outs:
  - data/raw/                  # Raw GDSC2 CSV files
  - data/splits/               # Train/val/test splits
```

**Output**: 
- `gdsc2_original.csv` - Original GDSC2 data
- `new_batch_2026.csv` - New incoming data
- `train.csv`, `val.csv`, `test.csv` - Split datasets

### Stage 2: Graph Building (`build_graph`)
```yaml
cmd: python src/preprocessing/build_graph.py
deps:
  - data/raw/
  - src/preprocessing/build_graph.py
outs:
  - data/processed/graph.pt    # PyTorch Geometric graph
```

**Output**: Heterogeneous knowledge graph for GNN

### Stage 3: Model Training (`train`)
```yaml
cmd: python src/training/train_gnn.py
deps:
  - data/processed/graph.pt/
  - data/splits/
  - src/training/train_gnn.py
params:
  - num_epochs, learning_rate, hidden_dim...
outs:
  - artifacts/model.pt         # Trained GNN weights
  - artifacts/metrics.json     # Evaluation metrics
```

**Outputs**: 
- Model artifact with full evaluation metrics
- Reproducible metrics for comparison

---

## 4. How to Use DVC

### 4.1 View Pipeline Status

```bash
# Show pipeline DAG
dvc dag

# Output:
#   +---+
#   |ingest|
#   +---+
#     |
#   +-----+
#   |build_graph|
#   +-----+
#     |
#   +-----+
#   |train|
#   +-----+
```

### 4.2 Run Full Pipeline

```bash
# Execute all stages (idempotent - only reruns changed stages)
dvc repro

# Output:
#   Running stage 'ingest'...
#   Running stage 'build_graph'...
#   Running stage 'train'...
#   All stages completed!
```

### 4.3 Run Specific Stage

```bash
# Rerun only training stage
dvc repro -s train

# Or restore a specific output
dvc checkout data/processed/graph.pt
```

### 4.4 View Metrics

```bash
# Show metrics from all runs
dvc metrics show

# Compare metrics between runs
dvc plots diff
```

### 4.5 Add New Data to Tracking

```bash
# Track a new artifact
dvc add data/raw/new_data.csv

# This creates: data/raw/new_data.csv.dvc (versioning file)
# Commit .dvc file to Git for tracking
git add data/raw/new_data.csv.dvc
git commit -m "Add new GDSC data version"
```

---

## 5. Version Control Workflow

### 5.1 Reproducible Runs

Each run is locked with exact dependencies:

```yaml
# dvc.lock (auto-generated, commit to Git)
schema: '2.0'
stages:
  ingest:
    cmd: python src/ingestion/data_ingest.py
    deps:
    - path: src/ingestion/data_ingest.py
      md5: abc123...
      size: 2048
    outs:
    - path: data/raw
      md5: def456...
      size: 51200
```

### 5.2 Reproducing Previous Results

```bash
# Clone repo
git clone <repo>

# Restore exact data/model from DVC
dvc pull

# Verify pipeline integrity
dvc status
# Output: Data and pipelines are up to date
```

### 5.3 Tracking Changes

```bash
# See what changed in pipeline
dvc diff

# Compare with specific commit
dvc diff abc123def

# Compare metrics between runs
dvc metrics diff
```

---

## 6. Remote Storage Setup

### 6.1 Configure S3 Remote (Example)

```bash
# Add AWS S3 as remote
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Set AWS credentials
export AWS_ACCESS_KEY_ID=xxx
export AWS_SECRET_ACCESS_KEY=yyy

# Push artifacts to S3
dvc push

# Pull artifacts from S3
dvc pull
```

### 6.2 Configure Local Remote (Testing)

```bash
# Create local storage directory
mkdir -p /mnt/dvc-storage

# Add as remote
dvc remote add local /mnt/dvc-storage
dvc remote default local

# Push to local storage
dvc push
```

### 6.3 View Remote Configuration

```bash
# Show all remotes
dvc remote list

# Show detailed config
cat .dvc/config
```

---

## 7. Continuous Integration with DVC

### 7.1 GitHub Actions Integration

Current workflow includes:

```yaml
# .github/workflows/retrain.yml
- name: Restore data and models
  run: dvc pull

- name: Run full pipeline
  run: dvc repro

- name: Push new artifacts
  run: dvc push
  if: github.ref == 'refs/heads/main'
```

### 7.2 Cache Optimization

DVC caches computed artifacts to avoid recomputation:

```bash
# View cache directory
ls -la .dvc/cache/

# Clean unused cache
dvc cache prune

# Remove all cache (fetch from remote again)
dvc cache remove -a
```

---

## 8. Best Practices

### 8.1 Do's ✅

- ✅ **Commit** `dvc.lock` to Git (enables reproducibility)
- ✅ **Push** data to remote regularly (`dvc push`)
- ✅ **Use** `dvc repro` for pipeline execution
- ✅ **Update** hyperparameters in `config.py`
- ✅ **Version** data with meaningful messages

### 8.2 Don'ts ❌

- ❌ **Don't** commit large files directly to Git (use DVC)
- ❌ **Don't** modify `.dvc` files manually
- ❌ **Don't** share AWS credentials in code
- ❌ **Don't** skip `dvc.lock` commits
- ❌ **Don't** run pipeline stages out of order

---

## 9. Troubleshooting

### Issue: "Could not find data at path"

```bash
# Solution: Restore data from remote
dvc pull
dvc checkout
```

### Issue: "Pipeline is out of date"

```bash
# Solution: Regenerate dvc.lock
dvc repro --force

# Or specific stage
dvc repro -s train --force
```

### Issue: "Cache corruption"

```bash
# Solution: Verify and fix cache
dvc cache dir <new-location>
dvc checkout --relink
```

---

## 10. Integration with MLflow

**DVC** handles data versioning  
**MLflow** handles experiment tracking

**Workflow**:
1. Data changes → DVC tracks and versions
2. Run experiment → MLflow logs metrics
3. Model trains → DVC stores model artifact
4. Each run is reproducible with both systems

```bash
# View MLflow experiments
mlflow ui --backend-store-uri ./mlruns

# Check DVC status
dvc status

# Both systems provide complete audit trail
```

---

## 11. Next Steps

### Immediate
- [x] DVC configuration initialized
- [x] Pipeline stages defined
- [x] Local storage configured
- [ ] Run `dvc repro` to validate

### Short-term
- [ ] Set up S3 remote for cloud backup
- [ ] Configure team access
- [ ] Document remote credentials

### Long-term
- [ ] Model registry integration
- [ ] Automated data quality checks
- [ ] Cost optimization for cloud storage

---

## 12. Resources

- **DVC Documentation**: https://dvc.org/doc
- **Getting Started**: https://dvc.org/doc/start
- **Pipeline Guide**: https://dvc.org/doc/user-guide/pipelines
- **Remote Storage**: https://dvc.org/doc/user-guide/remote
- **Best Practices**: https://dvc.org/doc/user-guide/best-practices

---

**Last Updated**: April 16, 2026  
**Next Review**: May 16, 2026
