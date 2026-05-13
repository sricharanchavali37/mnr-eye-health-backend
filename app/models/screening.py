import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, Boolean, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class RiskLevel(str, enum.Enum):
    low = "low"
    moderate = "moderate"
    high = "high"
    critical = "critical"


class Screening(Base):
    __tablename__ = "screenings"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)

    visual_acuity_left = Column(Float, nullable=True)
    visual_acuity_right = Column(Float, nullable=True)
    color_vision_score = Column(Integer, nullable=True)
    contrast_sensitivity = Column(Float, nullable=True)
    field_of_view_score = Column(Integer, nullable=True)

    symptoms = Column(Text, nullable=True)
    symptom_duration_days = Column(Integer, nullable=True)
    last_eye_exam_months = Column(Integer, nullable=True)

    age = Column(Integer, nullable=True)
    has_diabetes = Column(Boolean, default=False)
    has_hypertension = Column(Boolean, default=False)
    family_history = Column(Boolean, default=False)
    screen_time_hours = Column(Float, nullable=True)

    risk_level = Column(SAEnum(RiskLevel), nullable=True)
    risk_score = Column(Float, nullable=True)
    ai_recommendation = Column(Text, nullable=True)
    ai_summary = Column(Text, nullable=True)
    flags = Column(Text, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="screenings")
    report = relationship("HealthReport", back_populates="screening", uselist=False)
