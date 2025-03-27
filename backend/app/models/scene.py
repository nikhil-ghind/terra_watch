import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class Scene(Base):
    __tablename__ = "scenes"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=False)
    before_path = Column(String, nullable=False)
    after_path = Column(String, nullable=False)
    acquired_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    bbox_wkt = Column(String)
    status = Column(String, default="pending")
