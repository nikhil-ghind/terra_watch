import uuid
from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import DateTime, func
from app.database import Base


class Correction(Base):
    __tablename__ = "corrections"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detection_id = Column(UUID(as_uuid=True), ForeignKey("detections.id"), nullable=False)
    analyst_id = Column(String, nullable=False)
    corrected_class = Column(String, nullable=False)
    note = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
