from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.reminder import Reminder, ReminderStatus
from app.schemas.reminder import ReminderCreate, ReminderResponse

router = APIRouter(prefix="/reminders", tags=["Reminders"])


def _serialize(r: Reminder) -> ReminderResponse:
    return ReminderResponse(
        id=r.id,
        user_id=r.user_id,
        title=r.title,
        message=r.message or "",
        remind_at=r.remind_at.isoformat(),
        is_recurring=r.is_recurring,
        recurrence_days=r.recurrence_days,
        status=r.status.value if r.status else "pending",
        created_at=r.created_at.isoformat(),
    )


@router.post("", response_model=ReminderResponse, status_code=status.HTTP_201_CREATED)
def create_reminder(
    payload: ReminderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Make remind_at timezone-aware for comparison
    remind_at = payload.remind_at
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=timezone.utc)

    if remind_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="remind_at must be in the future")

    reminder = Reminder(
        user_id=current_user.id,
        title=payload.title,
        message=payload.message or "",
        remind_at=remind_at,
        is_recurring=payload.is_recurring,
        recurrence_days=payload.recurrence_days,
    )
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return _serialize(reminder)


@router.get("", response_model=List[ReminderResponse])
def list_reminders(
    status_filter: str = "pending",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Reminder).filter(Reminder.user_id == current_user.id)
    if status_filter in ("pending", "sent", "cancelled"):
        query = query.filter(Reminder.status == status_filter)
    reminders = query.order_by(Reminder.remind_at.asc()).all()
    return [_serialize(r) for r in reminders]


@router.delete("/{reminder_id}", status_code=204)
def cancel_reminder(
    reminder_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    reminder = db.query(Reminder).filter(Reminder.id == reminder_id).first()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")
    if reminder.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    reminder.status = ReminderStatus.cancelled
    db.commit()


@router.get("/due/now")
def get_due_reminders(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    now = datetime.now(timezone.utc)
    due = (
        db.query(Reminder)
        .filter(
            Reminder.user_id == current_user.id,
            Reminder.status == ReminderStatus.pending,
            Reminder.remind_at <= now,
        )
        .all()
    )

    results = []
    for r in due:
        results.append(_serialize(r))
        r.status = ReminderStatus.sent
        if r.is_recurring and r.recurrence_days:
            from datetime import timedelta
            r.remind_at = r.remind_at + timedelta(days=int(r.recurrence_days))
            r.status = ReminderStatus.pending

    db.commit()
    return {"due_reminders": results, "count": len(results)}
