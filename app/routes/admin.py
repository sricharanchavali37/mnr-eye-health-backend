from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.core.deps import require_role
from app.models.user import User, UserRole
from app.models.screening import Screening
from app.schemas.auth import UserResponse

router = APIRouter(prefix="/admin", tags=["Admin"])

admin_only = require_role(UserRole.admin)


@router.get("/users", response_model=List[UserResponse])
def list_all_users(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    _=Depends(admin_only),
):
    users = db.query(User).offset(skip).limit(limit).all()
    return [
        UserResponse(
            id=u.id, full_name=u.full_name, email=u.email,
            role=u.role.value, is_active=u.is_active,
            phone=u.phone, date_of_birth=u.date_of_birth,
            created_at=u.created_at.isoformat(),
        )
        for u in users
    ]


@router.get("/stats")
def platform_stats(
    db: Session = Depends(get_db),
    _=Depends(admin_only),
):
    total_users = db.query(func.count(User.id)).scalar()
    active_users = db.query(func.count(User.id)).filter(User.is_active == True).scalar()
    total_screenings = db.query(func.count(Screening.id)).scalar()

    risk_dist = (
        db.query(Screening.risk_level, func.count(Screening.id))
        .group_by(Screening.risk_level)
        .all()
    )

    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_screenings": total_screenings,
        "risk_distribution": {
            (level.value if level else "unknown"): count
            for level, count in risk_dist
        },
    }


@router.patch("/users/{user_id}/role")
def update_user_role(
    user_id: str,
    new_role: UserRole,
    db: Session = Depends(get_db),
    _=Depends(admin_only),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = new_role
    db.commit()
    return {"message": f"Role updated to {new_role.value}"}


@router.patch("/users/{user_id}/activate")
def toggle_user_active(
    user_id: str,
    active: bool,
    db: Session = Depends(get_db),
    _=Depends(admin_only),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_active = active
    db.commit()
    return {"message": f"User {'activated' if active else 'deactivated'}"}
