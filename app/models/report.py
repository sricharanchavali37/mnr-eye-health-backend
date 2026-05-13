import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.core.database import Base


class HealthReport(Base):
    __tablename__ = "health_reports"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    screening_id = Column(String, ForeignKey("screenings.id"), nullable=True)

    title = Column(String(200), nullable=False)
    summary = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    pdf_path = Column(String(300), nullable=True)
    report_data = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="reports")
    screening = relationship("Screening", back_populates="report")
