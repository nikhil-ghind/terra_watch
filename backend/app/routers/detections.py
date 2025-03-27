import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.detection import Detection
from app.schemas.detection import DetectionResponse

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("/", response_model=list[DetectionResponse])
def list_detections(
    scene_id: uuid.UUID = None,
    predicted_class: str = None,
    min_confidence: float = None,
    skip: int = 0, limit: int = 50,
    db: Session = Depends(get_db),
):
    q = db.query(Detection)
    if predicted_class:
        q = q.filter(Detection.predicted_class == predicted_class)
    if min_confidence is not None:
        q = q.filter(Detection.confidence >= min_confidence)
    return q.offset(skip).limit(limit).all()


@router.get("/{detection_id}", response_model=DetectionResponse)
def get_detection(detection_id: uuid.UUID, db: Session = Depends(get_db)):
    d = db.query(Detection).filter(Detection.id == detection_id).first()
    if not d:
        raise HTTPException(status_code=404, detail="Detection not found")
    return d
