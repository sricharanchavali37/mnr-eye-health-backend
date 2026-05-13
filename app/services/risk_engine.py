import json
from typing import List, Tuple, Optional
from app.models.screening import RiskLevel

HIGH_RISK_SYMPTOMS = {
    "sudden vision loss", "double vision", "flashes of light",
    "floaters", "curtain over vision", "severe eye pain",
    "halos around lights", "tunnel vision"
}
MODERATE_RISK_SYMPTOMS = {
    "blurry vision", "eye strain", "headaches", "sensitivity to light",
    "difficulty reading", "night vision problems", "eye fatigue",
    "redness", "dry eyes"
}


def _acuity_score(val: Optional[float]) -> float:
    if val is None:
        return 5
    if val >= 0.8:
        return 0
    elif val >= 0.5:
        return 10
    elif val >= 0.3:
        return 20
    else:
        return 25


def compute_risk(data: dict) -> Tuple[float, RiskLevel, List[str]]:
    score = 0.0
    flags = []

    left = data.get("visual_acuity_left")
    right = data.get("visual_acuity_right")
    worst_eye = min(
        v for v in [left, right] if v is not None
    ) if (left is not None or right is not None) else None

    acuity_pts = _acuity_score(worst_eye)
    score += acuity_pts
    if acuity_pts >= 20:
        flags.append("Significantly reduced visual acuity detected")
    elif acuity_pts >= 10:
        flags.append("Mild reduction in visual acuity")

    cvs = data.get("color_vision_score")
    if cvs is not None:
        if cvs < 60:
            score += 15
            flags.append("Color vision deficiency detected")
        elif cvs < 80:
            score += 5

    cs = data.get("contrast_sensitivity")
    if cs is not None and cs < 0.5:
        score += 10
        flags.append("Low contrast sensitivity")

    fov = data.get("field_of_view_score")
    if fov is not None:
        if fov < 60:
            score += 15
            flags.append("Peripheral vision loss indicated")
        elif fov < 80:
            score += 7

    symptoms_raw = data.get("symptoms") or []
    if isinstance(symptoms_raw, str):
        try:
            symptoms_raw = json.loads(symptoms_raw)
        except Exception:
            symptoms_raw = []
    symptoms_lower = {s.lower().strip() for s in symptoms_raw}

    high_match = symptoms_lower & HIGH_RISK_SYMPTOMS
    mod_match = symptoms_lower & MODERATE_RISK_SYMPTOMS

    if high_match:
        score += min(25, len(high_match) * 10)
        flags.append(f"High-risk symptoms reported: {', '.join(high_match)}")
    if mod_match:
        score += min(10, len(mod_match) * 3)

    duration = data.get("symptom_duration_days")
    if duration and duration > 30:
        score += 5
        flags.append("Symptoms persisting over 30 days")

    if data.get("has_diabetes"):
        score += 10
        flags.append("Diabetic patient — elevated risk of diabetic retinopathy")
    if data.get("has_hypertension"):
        score += 8
        flags.append("Hypertension — risk of hypertensive retinopathy")
    if data.get("family_history"):
        score += 5
        flags.append("Family history of eye conditions")

    age = data.get("age")
    if age:
        if age >= 65:
            score += 8
            flags.append("Age 65+ — higher risk of age-related macular degeneration")
        elif age >= 50:
            score += 4
        elif age <= 10:
            score += 3
            flags.append("Pediatric patient — amblyopia/refractive error screening advised")

    screen_time = data.get("screen_time_hours")
    if screen_time and screen_time > 8:
        score += 3
        flags.append("High screen time — digital eye strain risk")

    last_exam = data.get("last_eye_exam_months")
    if last_exam is None or last_exam > 24:
        score += 5
        flags.append("No eye exam in 2+ years — routine screening overdue")

    score = min(score, 100.0)

    if score < 20:
        level = RiskLevel.low
    elif score < 45:
        level = RiskLevel.moderate
    elif score < 70:
        level = RiskLevel.high
    else:
        level = RiskLevel.critical

    return round(score, 1), level, flags
