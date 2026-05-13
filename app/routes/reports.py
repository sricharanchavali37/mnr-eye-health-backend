import os
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User, UserRole
from app.models.report import HealthReport

router = APIRouter(prefix="/reports", tags=["Health Reports"])


class ReportItem(BaseModel):
    id: str
    title: str
    summary: Optional[str] = None
    has_pdf: bool
    pdf_path: Optional[str] = None
    report_data: Optional[str] = None
    created_at: str

    class Config:
        from_attributes = True


@router.get("", response_model=List[ReportItem])
def list_reports(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    reports = (
        db.query(HealthReport)
        .filter(HealthReport.user_id == current_user.id)
        .order_by(HealthReport.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        ReportItem(
            id=r.id,
            title=r.title,
            summary=r.summary,
            has_pdf=bool(r.pdf_path and os.path.exists(r.pdf_path)),
            pdf_path=r.pdf_path if (r.pdf_path and os.path.exists(r.pdf_path)) else None,
            report_data=r.report_data,
            created_at=r.created_at.isoformat(),
        )
        for r in reports
    ]


@router.get("/{report_id}/pdf")
def download_report_pdf(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    report = db.query(HealthReport).filter(HealthReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if report.user_id != current_user.id and current_user.role not in (UserRole.admin, UserRole.doctor):
        raise HTTPException(status_code=403, detail="Access denied")

    if not report.pdf_path or not os.path.exists(report.pdf_path):
        raise HTTPException(status_code=404, detail="PDF not yet generated. Please retry in a moment.")

    return FileResponse(
        path=report.pdf_path,
        media_type="application/pdf",
        filename=f"mnr_eye_report_{report_id[:8]}.pdf",
    )


@router.get("/{report_id}")
def get_report(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    report = db.query(HealthReport).filter(HealthReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    if report.user_id != current_user.id and current_user.role not in (UserRole.admin, UserRole.doctor):
        raise HTTPException(status_code=403, detail="Access denied")

    recommendations = []
    if report.recommendations:
        try:
            recommendations = json.loads(report.recommendations)
        except Exception:
            recommendations = [report.recommendations]

    report_data = {}
    if report.report_data:
        try:
            report_data = json.loads(report.report_data)
        except Exception:
            pass

    return {
        "id": report.id,
        "title": report.title,
        "summary": report.summary,
        "recommendations": recommendations,
        "has_pdf": bool(report.pdf_path and os.path.exists(report.pdf_path)),
        "pdf_path": report.pdf_path if (report.pdf_path and os.path.exists(report.pdf_path)) else None,
        "report_data": report_data,
        "created_at": report.created_at.isoformat(),
    }
