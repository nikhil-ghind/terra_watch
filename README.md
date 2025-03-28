# Terra Watch

Land-use change detection from multispectral satellite imagery using ResNet-50 + Random Forest ensemble, with FastAPI backend, PostgreSQL persistence, and a React analyst dashboard.

## Overview

- **ML Pipeline**: ResNet-50 (frozen early layers) + Random Forest late fusion on 2048-d embeddings; 5 land-use change classes
- **FastAPI Backend**: `/scenes` (ingest), `/detections` (results), `/corrections` (analyst feedback)
- **PostgreSQL**: Scenes, tiles, detections, corrections with UUID PKs and cascade deletes
- **React Dashboard**: Scene list, detection tile grid, label correction form
- **Docker Compose**: Postgres + backend service

## Tech Stack

Python 3.11 · PyTorch · scikit-learn · rasterio · FastAPI · PostgreSQL · SQLAlchemy · React 18 · Vite

## Quickstart

```bash
# Start database
docker compose up -d db

# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# ML training
cd .. && pip install -r ml/requirements.txt
python -m ml.src.train --data_csv ml/data/sample.csv --epochs 20

# Frontend
cd frontend && npm install && npm run dev
```

## Architecture

```
Satellite GeoTIFF pairs (before/after)
    → tile_scene() → 224×224 patches
    → ResNet-50 backbone (frozen) → 2048-d embeddings
    → Random Forest soft-vote ensemble
    → Detection: predicted_class + confidence
    → PostgreSQL: scenes → tiles → detections → corrections
    → FastAPI REST API
    → React Dashboard (scene list, detection grid, label editor)
```

## Classes

| ID | Label |
|----|-------|
| 0 | no_change |
| 1 | deforestation |
| 2 | urban_expansion |
| 3 | water_shift |
| 4 | agriculture |

## Tests

```bash
# Backend tests (pytest, async)
cd backend && pytest tests/ -v
```

The `backend/tests/` directory is currently a placeholder; a representative end-to-end check
against a running stack:

```bash
# Ingest a scene pair
curl -X POST http://localhost:8000/scenes -F "before=@tile_t0.tif" -F "after=@tile_t1.tif"

# Fetch detections and corrections for the scene
curl http://localhost:8000/detections?scene_id=<id>
curl -X POST http://localhost:8000/corrections \
  -H "Content-Type: application/json" \
  -d '{"detection_id":"<id>","corrected_class":1}'
```

## Evaluation

This is a 5-class classification problem (`no_change`, `deforestation`, `urban_expansion`,
`water_shift`, `agriculture`). Training (`ml/src/train.py`) computes:

- **Macro F1** on the held-out validation split via `sklearn.metrics.f1_score(..., average="macro")`
  after each epoch; the best checkpoint is retained.
- **Per-epoch loss** (cross-entropy) for both train and val splits.

Recommended additional offline metrics to compute on the test split:

| Metric | How to compute |
|--------|----------------|
| Per-class precision / recall / F1 | `sklearn.metrics.classification_report` over predictions vs. labels |
| Confusion matrix | `sklearn.metrics.confusion_matrix`, normalized rows |
| Macro and weighted ROC-AUC | `roc_auc_score(y_true, y_score, multi_class="ovr")` on RF soft-vote probabilities |
| Calibration | reliability diagram of predicted confidence vs. empirical accuracy (binned) |

The data split is defined by `ml/data/sample.csv` (`split` column: `train`/`val`/`test`). Run
`python -m ml.src.train --data_csv ml/data/sample.csv --epochs 20` to reproduce the headline
macro-F1 number printed at the end of training.
