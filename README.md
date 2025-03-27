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
