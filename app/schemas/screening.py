from pydantic import BaseModel, Field
from typing import Optional, List


class ScreeningRequest(BaseModel):
    visual_acuity_left: Optional[float] = Field(None, ge=0.0, le=2.0)
    visual_acuity_right: Optional[float] = Field(None, ge=0.0, le=2.0)
    color_vision_score: Optional[int] = Field(None, ge=0, le=100)
    contrast_sensitivity: Optional[float] = Field(None, ge=0.0, le=2.0)
    field_of_view_score: Optional[int] = Field(None, ge=0, le=100)

    symptoms: Optional[List[str]] = Field(default_factory=list)
    symptom_duration_days: Optional[int] = Field(None, ge=0)
    last_eye_exam_months: Optional[int] = Field(None, ge=0)

    age: Optional[int] = Field(None, ge=1, le=120)
    has_diabetes: bool = False
    has_hypertension: bool = False
    family_history: bool = False
    screen_time_hours: Optional[float] = Field(None, ge=0.0, le=24.0)


class ScreeningResponse(BaseModel):
    id: str
    user_id: str
    risk_level: Optional[str]
    risk_score: Optional[float]
    ai_recommendation: Optional[str]
    ai_summary: Optional[str]
    flags: Optional[List[str]]
    created_at: str

    class Config:
        from_attributes = True
        use_enum_values = True


class ScreeningListItem(BaseModel):
    id: str
    risk_level: Optional[str]
    risk_score: Optional[float]
    created_at: str

    class Config:
        from_attributes = True
        use_enum_values = True
