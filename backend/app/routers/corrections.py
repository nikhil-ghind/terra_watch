import uuid
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.correction import Correction
from app.schemas.correction import CorrectionCreate, CorrectionResponse

router = APIRouter(prefix="/corrections", tags=["corrections"])


@router.post("/", response_model=CorrectionResponse)
def create_correction(body: CorrectionCreate, db: Session = Depends(get_db)):
    c = Correction(**body.model_dump())
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@router.get("/", response_model=list[CorrectionResponse])
def list_corrections(
    detection_id: uuid.UUID = None,
    analyst_id: str = None,
    db: Session = Depends(get_db),
):
    q = db.query(Correction)
    if detection_id:
        q = q.filter(Correction.detection_id == detection_id)
    if analyst_id:
        q = q.filter(Correction.analyst_id == analyst_id)
    return q.all()
