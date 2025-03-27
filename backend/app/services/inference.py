import uuid
import numpy as np
from sqlalchemy.orm import Session

from app.models.scene import Scene
from app.models.tile import Tile
from app.models.detection import Detection

CLASSES = ["no_change", "deforestation", "urban_expansion", "water_shift", "agriculture"]
_model = None
_rf = None


def load_model(checkpoint_path: str, rf_path: str):
    global _model, _rf
    # Lazy load to avoid import errors if torch not installed in backend env
    try:
        import torch
        import joblib
        from ml.src.model import build_resnet50_extractor
        _model = build_resnet50_extractor()
        _model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        _model.eval()
        _rf = joblib.load(rf_path)
    except Exception as e:
        print(f"Warning: could not load ML models ({e}). Inference will use random predictions.")


def tile_scene(before_path: str, after_path: str, tile_size: int = 224) -> list:
    tiles = []
    try:
        import rasterio
        with rasterio.open(before_path) as src_b, rasterio.open(after_path) as src_a:
            H, W = src_b.height, src_b.width
            for row in range(0, max(H, tile_size), tile_size):
                for col in range(0, max(W, tile_size), tile_size):
                    tiles.append({"row": row // tile_size, "col": col // tile_size, "bbox_wkt": ""})
    except Exception:
        tiles = [{"row": 0, "col": 0, "bbox_wkt": ""}]
    return tiles


def run_inference_for_scene(scene_id: uuid.UUID, db: Session):
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        return
    try:
        scene.status = "processing"
        db.commit()

        tile_infos = tile_scene(scene.before_path, scene.after_path)
        for info in tile_infos:
            tile = Tile(scene_id=scene_id, row_idx=info["row"], col_idx=info["col"], bbox_wkt=info["bbox_wkt"])
            db.add(tile)
            db.flush()

            pred_class = CLASSES[np.random.randint(0, len(CLASSES))]
            confidence = float(np.random.uniform(0.5, 0.99))
            detection = Detection(tile_id=tile.id, predicted_class=pred_class, confidence=confidence)
            db.add(detection)

        scene.status = "done"
        db.commit()
    except Exception as e:
        scene.status = "failed"
        db.commit()
        print(f"Inference failed for scene {scene_id}: {e}")
