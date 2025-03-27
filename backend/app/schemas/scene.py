from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Optional


class SceneCreate(BaseModel):
    name: str
    acquired_at: datetime


class SceneListItem(BaseModel):
    id: UUID
    name: str
    status: str
    acquired_at: datetime
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class SceneResponse(SceneListItem):
    before_path: str
    after_path: str
    bbox_wkt: Optional[str] = None
