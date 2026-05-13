from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ReminderCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    message: Optional[str] = Field(default="", max_length=500)
    remind_at: datetime
    is_recurring: bool = False
    recurrence_days: Optional[int] = Field(None, ge=1, le=365)


class ReminderResponse(BaseModel):
    id: str
    user_id: str
    title: str
    message: str
    remind_at: str
    is_recurring: bool
    recurrence_days: Optional[int] = None
    status: str
    created_at: str

    class Config:
        from_attributes = True
        use_enum_values = True
