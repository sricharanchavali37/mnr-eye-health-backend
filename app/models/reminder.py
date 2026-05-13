import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Integer, Enum as SAEnum
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class ReminderStatus(str, enum.Enum):
    pending = "pending"
    sent = "sent"
    cancelled = "cancelled"


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(String(500), nullable=False, default="")
    remind_at = Column(DateTime, nullable=False)
    is_recurring = Column(Boolean, default=False)
    recurrence_days = Column(Integer, nullable=True)
    status = Column(SAEnum(ReminderStatus), default=ReminderStatus.pending)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="reminders")
