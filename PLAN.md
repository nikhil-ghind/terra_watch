# Terra Watch

## Project Overview

A proof-of-concept system for classifying land-use changes in multispectral satellite imagery. The system ingests raw satellite scenes, runs a deep learning classification pipeline (ResNet-50 + Random Forest ensemble), exposes results via a REST API, persists detections in PostgreSQL, and serves an analyst-facing React dashboard for reviewing and correcting classification labels.

Key goals:
- Classify land-use change categories (deforestation, urban expansion, water body shift, agricultural conversion) from before/after multispectral image pairs.
- Achieve >85% macro F1 on a held-out validation set.
- Enable analyst correction workflows so ground-truth labels accumulate over time.
- Keep inference latency under 2 seconds per image tile on CPU.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | PyTorch 2.2, torchvision 0.17 |
| CNN backbone | ResNet-50 (pretrained on ImageNet, fine-tuned) |
| Classical ML | scikit-learn 1.4 (Random Forest, preprocessing) |
| API server | FastAPI 0.110, Uvicorn 0.29 |
| Database | PostgreSQL 16, psycopg2 2.9, Alembic 1.13 |
| Frontend | React 18, Vite 5, Axios, Leaflet.js |
| Image I/O | rasterio 1.3, numpy 1.26, Pillow 10 |
| Data management | pandas 2.2, scikit-image 0.22 |
| Testing | pytest 8, httpx (async FastAPI testing) |
| Environment | Python 3.11, Node 20, Docker Compose |

---

## Architecture Overview

```
[Satellite imagery (GeoTIFF)] --> [Preprocessing Service]
        |
        v
[Feature Extractor: ResNet-50 backbone (frozen conv layers)]
        |
        v
[Ensemble Layer: ResNet-50 head + Random Forest on extracted features]
        |
        v
[Prediction Store: PostgreSQL]
        |
        v
[FastAPI REST API] <--> [React Dashboard (analyst review + label correction)]
```

Components:
- **Preprocessing Service**: Loads multispectral GeoTIFFs, normalizes bands, generates change difference tensors from before/after pairs.
- **Feature Extractor**: ResNet-50 with frozen early layers; outputs 2048-d embeddings per tile.
- **Classifier Head**: Fine-tuned fully connected layers on ResNet + a parallel Random Forest trained on the same embeddings (late fusion via soft voting).
- **FastAPI Backend**: Exposes `/predict`, `/detections`, `/corrections` endpoints; manages job queue.
- **PostgreSQL**: Stores scene metadata, tile detections, analyst corrections, and audit log.
- **React Dashboard**: Map view (Leaflet), detection list, label editor, confidence histogram.

---

## Phase 1: Project Scaffolding and Environment Setup

**Goal**: Establish a reproducible project structure, virtual environments, and database connectivity.

### Tasks

1. Create the top-level directory layout:
   ```
   geospatialChangeDetection/
   ├── backend/
   │   ├── app/
   │   │   ├── __init__.py
   │   │   ├── main.py
   │   │   ├── config.py
   │   │   ├── database.py
   │   │   ├── models/
   │   │   ├── routers/
   │   │   └── services/
   │   ├── alembic/
   │   ├── alembic.ini
   │   ├── requirements.txt
   │   └── tests/
   ├── ml/
   │   ├── data/
   │   ├── notebooks/
   │   ├── src/
   │   │   ├── __init__.py
   │   │   ├── dataset.py
   │   │   ├── model.py
   │   │   ├── train.py
   │   │   ├── evaluate.py
   │   │   └── predict.py
   │   └── requirements.txt
   ├── frontend/
   │   ├── src/
   │   │   ├── components/
   │   │   ├── pages/
   │   │   └── api/
   │   ├── package.json
   │   └── vite.config.js
   └── docker-compose.yml
   ```

2. Create `backend/requirements.txt`:
   ```
   fastapi==0.110.0
   uvicorn[standard]==0.29.0
   psycopg2-binary==2.9.9
   sqlalchemy==2.0.29
   alembic==1.13.1
   pydantic==2.6.4
   pydantic-settings==2.2.1
   httpx==0.27.0
   pytest==8.1.1
   pytest-asyncio==0.23.6
   rasterio==1.3.9
   numpy==1.26.4
   Pillow==10.2.0
   python-multipart==0.0.9
   ```

3. Create `ml/requirements.txt`:
   ```
   torch==2.2.2
   torchvision==0.17.2
   scikit-learn==1.4.1
   pandas==2.2.1
   scikit-image==0.22.0
   rasterio==1.3.9
   numpy==1.26.4
   matplotlib==3.8.3
   tqdm==4.66.2
   joblib==1.3.2
   ```

4. Create `docker-compose.yml` defining two services:
   - `db`: image `postgres:16-alpine`, env vars `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB=geodetect`, volume `pgdata:/var/lib/postgresql/data`, port `5432:5432`.
   - `backend`: build from `./backend`, depends_on `db`, env file `.env`, port `8000:8000`, command `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.

5. Create `backend/app/config.py` using `pydantic_settings.BaseSettings` with fields: `DATABASE_URL: str`, `SECRET_KEY: str`, `DEBUG: bool = False`. Load from environment.

6. Create `.env.example` with placeholder values for all required environment variables.

7. Run `pip install -r backend/requirements.txt` and `pip install -r ml/requirements.txt` in separate virtual environments.

8. Run `docker compose up -d db` and verify PostgreSQL is reachable via `psql`.

---

## Phase 2: Database Schema and Migrations

**Goal**: Define and migrate the full PostgreSQL schema covering scenes, tiles, detections, and analyst corrections.

### Tasks

1. Initialize Alembic inside `backend/`:
   ```
   cd backend && alembic init alembic
   ```

2. Edit `alembic/env.py` to import `Base` from `app.database` and set `target_metadata = Base.metadata`.

3. Create `backend/app/database.py`:
   - Define SQLAlchemy `engine` using `DATABASE_URL` from config.
   - Define `SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)`.
   - Define `Base = declarative_base()`.
   - Define `get_db()` dependency that yields a session and closes it.

4. Create `backend/app/models/scene.py` — SQLAlchemy ORM model `Scene`:
   - `id: UUID` (primary key, default `uuid4`)
   - `name: str` (unique, not null)
   - `before_path: str`
   - `after_path: str`
   - `acquired_at: datetime`
   - `created_at: datetime` (server default `now()`)
   - `bbox_wkt: str` (WKT polygon string)
   - `status: str` (enum: `pending`, `processing`, `done`, `failed`; default `pending`)

5. Create `backend/app/models/tile.py` — ORM model `Tile`:
   - `id: UUID`
   - `scene_id: UUID` (ForeignKey `scenes.id`, on delete CASCADE)
   - `row_idx: int`, `col_idx: int`
   - `bbox_wkt: str`
   - `before_patch_path: str`
   - `after_patch_path: str`

6. Create `backend/app/models/detection.py` — ORM model `Detection`:
   - `id: UUID`
   - `tile_id: UUID` (ForeignKey `tiles.id`)
   - `predicted_class: str`
   - `confidence: float`
   - `embedding: list[float]` (stored as `ARRAY(Float)` via `postgresql.ARRAY`)
   - `created_at: datetime`

7. Create `backend/app/models/correction.py` — ORM model `Correction`:
   - `id: UUID`
   - `detection_id: UUID` (ForeignKey `detections.id`)
   - `analyst_id: str`
   - `corrected_class: str`
   - `note: str` (nullable)
   - `created_at: datetime`

8. Create `backend/app/models/__init__.py` importing all models so Alembic discovers them.

9. Generate and apply initial migration:
   ```
   alembic revision --autogenerate -m "initial_schema"
   alembic upgrade head
   ```

10. Write `backend/tests/test_db.py`: a pytest test that creates a `Scene` row, queries it back, and asserts field equality using a test database session.

---

## Phase 3: ML Pipeline — Data Preparation and Feature Extraction

**Goal**: Build the dataset loader, preprocessing pipeline, and ResNet-50 feature extractor.

### Tasks

1. Create `ml/src/dataset.py` — class `ChangeDetectionDataset(torch.utils.data.Dataset)`:
   - `__init__(self, csv_path: str, transform=None)`: reads a CSV with columns `before_path`, `after_path`, `label`.
   - `__len__`: returns number of rows.
   - `__getitem__(self, idx)`: loads both GeoTIFF patches using `rasterio`, selects RGB bands (indices 0, 1, 2) or all 4 for multispectral, normalizes to `[0, 1]`, computes per-pixel absolute difference tensor `(C, H, W)`, stacks `[before, after, diff]` along channel dim → `(3C, H, W)`, returns `(tensor, label_int)`.
   - Define `LABEL_TO_INT: dict[str, int]` mapping class names to integers at module level.
   - Define `get_transforms(train: bool)` returning `torchvision.transforms.Compose` with: resize to `(224, 224)`, optional random horizontal/vertical flip and color jitter (train only), normalize with ImageNet mean/std.

2. Create `ml/src/model.py`:
   - Function `build_resnet50_extractor() -> nn.Module`: loads `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`, freezes all layers except `layer3`, `layer4`, and `fc`. Replaces `fc` with `nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))`. Returns model.
   - Function `build_embedding_extractor(model: nn.Module) -> nn.Module`: wraps the model up to `avgpool` (excluding `fc`) to extract 2048-d embeddings. Uses `nn.Sequential(*list(model.children())[:-1])` and adds `nn.Flatten()`.
   - Constant `NUM_CLASSES = 5` (no_change, deforestation, urban_expansion, water_shift, agriculture).

3. Create `ml/src/train.py`:
   - `train_one_epoch(model, loader, optimizer, criterion, device) -> float`: standard training loop, returns average loss.
   - `validate(model, loader, criterion, device) -> tuple[float, float]`: returns avg loss and macro F1 score using `sklearn.metrics.f1_score`.
   - `main()` function:
     - Parse args: `--data_csv`, `--epochs` (default 20), `--batch_size` (default 32), `--lr` (default 1e-4), `--output_dir`.
     - Build dataset, split 80/20 train/val using `torch.utils.data.random_split`.
     - Instantiate `DataLoader` for each split (num_workers=4, pin_memory=True).
     - Build model via `build_resnet50_extractor()`, move to `device`.
     - Optimizer: `Adam` with `lr`, weight decay `1e-5`.
     - Scheduler: `CosineAnnealingLR(optimizer, T_max=epochs)`.
     - Criterion: `CrossEntropyLoss`.
     - Training loop: run `train_one_epoch` and `validate`, print metrics, save best model checkpoint to `{output_dir}/best_model.pt` when val F1 improves.
     - Save final model to `{output_dir}/final_model.pt`.

4. Create `ml/src/evaluate.py`:
   - `generate_embeddings(extractor, loader, device) -> tuple[np.ndarray, np.ndarray]`: returns `(embeddings, labels)` arrays.
   - `train_random_forest(embeddings, labels) -> RandomForestClassifier`: fits `RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)`.
   - `ensemble_predict(resnet_probs, rf_probs, alpha=0.6) -> np.ndarray`: soft-vote ensemble: `alpha * resnet_probs + (1-alpha) * rf_probs`, returns argmax.
   - `evaluate_ensemble(model, rf, loader, device) -> dict`: returns dict with `macro_f1`, `accuracy`, `per_class_f1`, `confusion_matrix`.
   - `main()`: load best checkpoint, generate embeddings on val split, train RF, run ensemble evaluation, print classification report, save RF model via `joblib.dump`.

5. Create `ml/data/README_data_format.txt` describing expected CSV schema and GeoTIFF band ordering.

6. Run training on a small synthetic dataset (5 scenes, 50 tiles) to verify the pipeline runs end-to-end without errors:
   ```
   python ml/src/train.py --data_csv ml/data/sample.csv --epochs 2 --output_dir ml/checkpoints/
   ```

---

## Phase 4: FastAPI Backend

**Goal**: Implement all REST endpoints for scene ingestion, detection retrieval, and analyst correction.

### Tasks

1. Create `backend/app/main.py`:
   - Initialize `FastAPI(title="GeoDetect API", version="1.0")`.
   - Include routers from `routers/scenes.py`, `routers/detections.py`, `routers/corrections.py`.
   - Add CORS middleware allowing `http://localhost:5173` (React dev server).
   - Add startup event: log database URL (masked).

2. Create `backend/app/routers/scenes.py` with `APIRouter(prefix="/scenes", tags=["scenes"])`:
   - `POST /scenes/` — accepts multipart upload of `before_file` and `after_file` (GeoTIFF), `name: str`, `acquired_at: datetime`. Saves files to `UPLOAD_DIR/{scene_id}/`. Inserts `Scene` row with `status=pending`. Returns scene JSON. Triggers background task `services.inference.run_inference_for_scene(scene_id)`.
   - `GET /scenes/` — paginated list (query params: `skip=0`, `limit=20`). Returns list of scene summaries.
   - `GET /scenes/{scene_id}` — returns full scene including associated tile count and detection count.
   - `DELETE /scenes/{scene_id}` — deletes scene and cascades.

3. Create `backend/app/routers/detections.py` with `APIRouter(prefix="/detections", tags=["detections"])`:
   - `GET /detections/` — filter by `scene_id`, `predicted_class`, `min_confidence` (float). Returns paginated detection list.
   - `GET /detections/{detection_id}` — full detection record including tile info.

4. Create `backend/app/routers/corrections.py` with `APIRouter(prefix="/corrections", tags=["corrections"])`:
   - `POST /corrections/` — body: `detection_id: UUID`, `analyst_id: str`, `corrected_class: str`, `note: str | None`. Inserts `Correction` row. Returns correction JSON.
   - `GET /corrections/` — filter by `detection_id` or `analyst_id`.

5. Create `backend/app/services/inference.py`:
   - `load_model(checkpoint_path: str, rf_path: str)`: loads ResNet-50 checkpoint and RF joblib file. Caches in module-level variables.
   - `tile_scene(before_path: str, after_path: str, tile_size: int = 224) -> list[dict]`: opens rasters with rasterio, generates a grid of `tile_size x tile_size` patches, returns list of `{row, col, before_patch: np.ndarray, after_patch: np.ndarray, bbox_wkt}`.
   - `run_inference_for_scene(scene_id: UUID, db: Session)`: fetches scene, tiles it, runs model + RF ensemble on each tile, inserts `Tile` and `Detection` rows, updates scene status to `done` (or `failed` on exception).

6. Create `backend/app/schemas/` with Pydantic v2 `BaseModel` classes:
   - `scene.py`: `SceneCreate`, `SceneResponse`, `SceneListItem`.
   - `detection.py`: `DetectionResponse`.
   - `correction.py`: `CorrectionCreate`, `CorrectionResponse`.

7. Create `backend/tests/test_api.py` using `httpx.AsyncClient` and `pytest-asyncio`:
   - Test `POST /scenes/` with a synthetic 224x224 3-band GeoTIFF.
   - Test `GET /scenes/` returns 200 with a list.
   - Test `POST /corrections/` with a valid `detection_id`.
   - Use a separate test PostgreSQL database URL set via environment variable.

8. Start the server and verify via curl:
   ```
   uvicorn app.main:app --reload --port 8000
   curl http://localhost:8000/docs
   ```

---

## Phase 5: React Dashboard

**Goal**: Build an analyst-facing web UI showing detections on a map, allowing label inspection and correction.

### Tasks

1. Scaffold the React project:
   ```
   npm create vite@5 frontend -- --template react
   cd frontend && npm install
   npm install axios leaflet react-leaflet @radix-ui/react-dialog recharts
   ```

2. Configure `frontend/vite.config.js` with a dev proxy:
   ```js
   server: { proxy: { '/api': { target: 'http://localhost:8000', rewrite: path => path.replace(/^\/api/, '') } } }
   ```

3. Create `frontend/src/api/client.js`: Axios instance with `baseURL: '/api'`, default headers `Content-Type: application/json`.

4. Create `frontend/src/api/scenes.js`: functions `fetchScenes()`, `uploadScene(formData)`, `fetchScene(id)`.

5. Create `frontend/src/api/detections.js`: functions `fetchDetections(params)`, `fetchDetection(id)`.

6. Create `frontend/src/api/corrections.js`: function `submitCorrection(payload)`.

7. Create `frontend/src/pages/Dashboard.jsx`:
   - Layout: left panel (scene list), center panel (map), right panel (detection inspector).
   - Fetches scene list on mount via `fetchScenes()`.
   - Selected scene triggers fetch of its detections.

8. Create `frontend/src/components/SceneList.jsx`:
   - Renders a scrollable list of scenes with name, status badge, and acquired date.
   - Calls `onSelect(sceneId)` on click.

9. Create `frontend/src/components/DetectionMap.jsx`:
   - Uses `react-leaflet` `MapContainer` + `TileLayer` (OpenStreetMap).
   - Renders each detection tile as a `Rectangle` with color coded by predicted class.
   - On click, calls `onSelectDetection(detection)`.

10. Create `frontend/src/components/DetectionInspector.jsx`:
    - Shows detection ID, predicted class, confidence score, tile coordinates.
    - Shows existing correction if any.
    - Renders a `<select>` dropdown with all 5 class names and a text input for analyst note.
    - On submit, calls `submitCorrection({ detection_id, analyst_id, corrected_class, note })` and shows success toast.

11. Create `frontend/src/components/ConfidenceHistogram.jsx`:
    - Uses `recharts` `BarChart` to display distribution of confidence scores for the selected scene's detections.

12. Run and verify:
    ```
    cd frontend && npm run dev
    ```
    Confirm dashboard loads, scenes are listed, map renders tiles, and correction form submits.

---

## Phase 6: Integration Testing and Deployment Hardening

**Goal**: End-to-end tests, performance validation, and production-ready configuration.

### Tasks

1. Write `backend/tests/test_integration.py`:
   - Upload a real small GeoTIFF scene pair, poll `GET /scenes/{id}` until status is `done`, verify detections exist, submit a correction, verify correction is retrievable.

2. Validate inference latency: run `time python ml/src/predict.py --tile path/to/tile.tif` and assert it completes in under 2 seconds on CPU.

3. Benchmark macro F1 on held-out test split: run `ml/src/evaluate.py --test_csv ml/data/test.csv` and assert F1 > 0.85. Document result in `ml/results/eval_report.txt`.

4. Add `backend/app/middleware/error_handler.py`: global exception handler returning `{"detail": str(exc)}` with status 500 and logging stack trace.

5. Update `docker-compose.yml` to add backend service health check: `test: ["CMD", "curl", "-f", "http://localhost:8000/health"]`, `interval: 30s`, `retries: 3`.

6. Add `GET /health` endpoint in `backend/app/main.py` returning `{"status": "ok", "db": "connected"}` after executing `SELECT 1` via the session.

7. Add database connection pooling parameters to the SQLAlchemy engine: `pool_size=10`, `max_overflow=20`, `pool_pre_ping=True`.

8. Build and test the full Docker Compose stack:
   ```
   docker compose up --build
   curl http://localhost:8000/health
   ```

9. Run the full pytest suite:
   ```
   pytest backend/tests/ -v --tb=short
   ```
   All tests must pass.
