from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional


class CorrectionCreate(BaseModel):
    detection_id: UUID
    analyst_id: str
    corrected_class: str
    note: Optional[str] = None


class CorrectionResponse(CorrectionCreate):
    id: UUID
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
