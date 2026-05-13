import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User, UserRole
from app.models.screening import Screening
from app.models.report import HealthReport
from app.schemas.screening import ScreeningRequest, ScreeningResponse, ScreeningListItem
from app.services.risk_engine import compute_risk
from app.services.ai_service import get_ai_analysis
from app.services.pdf_service import generate_report_pdf

router = APIRouter(prefix="/screenings", tags=["Vision Screening"])


def _serialize_screening(s: Screening) -> ScreeningResponse:
    flags = []
    if s.flags:
        try:
            flags = json.loads(s.flags)
        except Exception:
            flags = [s.flags]
    return ScreeningResponse(
        id=s.id,
        user_id=s.user_id,
        risk_level=s.risk_level.value if s.risk_level else None,
        risk_score=s.risk_score,
        ai_recommendation=s.ai_recommendation,
        ai_summary=s.ai_summary,
        flags=flags,
        created_at=s.created_at.isoformat(),
    )


@router.post("", response_model=ScreeningResponse, status_code=status.HTTP_201_CREATED)
async def submit_screening(
    payload: ScreeningRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    data = payload.model_dump()

    risk_score, risk_level, flags = compute_risk(data)

    ai_result = await get_ai_analysis(
        screening_data=data,
        risk_score=risk_score,
        risk_level=risk_level.value,
        flags=flags
    )

    screening = Screening(
        user_id=current_user.id,
        visual_acuity_left=payload.visual_acuity_left,
        visual_acuity_right=payload.visual_acuity_right,
        color_vision_score=payload.color_vision_score,
        contrast_sensitivity=payload.contrast_sensitivity,
        field_of_view_score=payload.field_of_view_score,
        symptoms=json.dumps(payload.symptoms or []),
        symptom_duration_days=payload.symptom_duration_days,
        last_eye_exam_months=payload.last_eye_exam_months,
        age=payload.age,
        has_diabetes=payload.has_diabetes,
        has_hypertension=payload.has_hypertension,
        family_history=payload.family_history,
        screen_time_hours=payload.screen_time_hours,
        risk_level=risk_level,
        risk_score=risk_score,
        ai_recommendation=json.dumps(ai_result.get("recommendations", [])),
        ai_summary=ai_result.get("summary", ""),
        flags=json.dumps(flags),
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)

    # Generate PDF report in background
    background_tasks.add_task(
        _create_report_background,
        screening_id=screening.id,
        user_name=current_user.full_name,
        user_email=current_user.email,
        data=data,
        risk_score=risk_score,
        risk_level=risk_level.value,
        flags=flags,
        ai_result=ai_result,
    )

    return _serialize_screening(screening)


def _create_report_background(screening_id, user_name, user_email,
                               data, risk_score, risk_level, flags, ai_result):
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        screening = db.query(Screening).filter(Screening.id == screening_id).first()
        if not screening:
            return

        pdf_path = None
        try:
            recommendations = ai_result.get("recommendations", [])
            pdf_path = generate_report_pdf(
                report_id=screening_id,
                user_name=user_name,
                user_email=user_email,
                screening_data=data,
                risk_score=risk_score,
                risk_level=risk_level,
                flags=flags,
                ai_summary=ai_result.get("summary", ""),
                recommendations=recommendations,
                urgent_referral=ai_result.get("urgent_referral", False),
                referral_reason=ai_result.get("referral_reason", ""),
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"PDF generation error: {e}")

        report = HealthReport(
            user_id=screening.user_id,
            screening_id=screening_id,
            title=f"Vision Screening Report — {risk_level.capitalize()} Risk",
            summary=ai_result.get("summary", ""),
            recommendations=json.dumps(ai_result.get("recommendations", [])),
            pdf_path=pdf_path,
            report_data=json.dumps({
                "risk_score": risk_score,
                "risk_level": risk_level,
                "flags": flags,
                "ai_result": ai_result,
                "screening_data": data,
            }),
        )
        db.add(report)
        db.commit()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Report creation error: {e}")
    finally:
        db.close()


@router.get("", response_model=List[ScreeningListItem])
def list_screenings(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    screenings = (
        db.query(Screening)
        .filter(Screening.user_id == current_user.id)
        .order_by(Screening.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        ScreeningListItem(
            id=s.id,
            risk_level=s.risk_level.value if s.risk_level else None,
            risk_score=s.risk_score,
            created_at=s.created_at.isoformat(),
        )
        for s in screenings
    ]


@router.get("/{screening_id}", response_model=ScreeningResponse)
def get_screening(
    screening_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    if screening.user_id != current_user.id and current_user.role not in (UserRole.admin, UserRole.doctor):
        raise HTTPException(status_code=403, detail="Access denied")
    return _serialize_screening(screening)


@router.delete("/{screening_id}", status_code=204)
def delete_screening(
    screening_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    screening = db.query(Screening).filter(Screening.id == screening_id).first()
    if not screening:
        raise HTTPException(status_code=404, detail="Screening not found")
    if screening.user_id != current_user.id and current_user.role != UserRole.admin:
        raise HTTPException(status_code=403, detail="Access denied")
    db.delete(screening)
    db.commit()
