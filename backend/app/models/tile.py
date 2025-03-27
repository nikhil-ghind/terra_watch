import uuid
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base


class Tile(Base):
    __tablename__ = "tiles"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scene_id = Column(UUID(as_uuid=True), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False)
    row_idx = Column(Integer)
    col_idx = Column(Integer)
    bbox_wkt = Column(String)
    before_patch_path = Column(String)
    after_patch_path = Column(String)
