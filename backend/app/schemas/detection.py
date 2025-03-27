from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Optional


class DetectionResponse(BaseModel):
    id: UUID
    tile_id: UUID
    predicted_class: str
    confidence: float
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True
