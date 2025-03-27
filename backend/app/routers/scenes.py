import os
import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.scene import Scene
from app.schemas.scene import SceneResponse, SceneListItem
from app.config import settings

router = APIRouter(prefix="/scenes", tags=["scenes"])


def _save_file(upload: UploadFile, scene_dir: str, name: str) -> str:
    os.makedirs(scene_dir, exist_ok=True)
    path = os.path.join(scene_dir, name)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path


@router.post("/", response_model=SceneResponse)
async def create_scene(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    acquired_at: datetime = Form(...),
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    scene_id = uuid.uuid4()
    scene_dir = os.path.join(settings.UPLOAD_DIR, str(scene_id))
    before_path = _save_file(before_file, scene_dir, "before.tif")
    after_path = _save_file(after_file, scene_dir, "after.tif")

    scene = Scene(id=scene_id, name=name, before_path=before_path, after_path=after_path,
                  acquired_at=acquired_at, status="pending")
    db.add(scene)
    db.commit()
    db.refresh(scene)

    from app.services import inference as infer_svc
    background_tasks.add_task(infer_svc.run_inference_for_scene, scene_id, db)
    return scene


@router.get("/", response_model=list[SceneListItem])
def list_scenes(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return db.query(Scene).offset(skip).limit(limit).all()


@router.get("/{scene_id}", response_model=SceneResponse)
def get_scene(scene_id: uuid.UUID, db: Session = Depends(get_db)):
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return scene


@router.delete("/{scene_id}")
def delete_scene(scene_id: uuid.UUID, db: Session = Depends(get_db)):
    scene = db.query(Scene).filter(Scene.id == scene_id).first()
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    db.delete(scene)
    db.commit()
    return {"detail": "deleted"}
