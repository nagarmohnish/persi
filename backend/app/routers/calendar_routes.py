from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app.database import get_db
from app.config import get_settings
from app.models.calendar import CalendarEvent
from app.schemas.calendar import CalendarEventCreate, CalendarEventUpdate, CalendarEventOut

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


@router.get("/events/", response_model=list[CalendarEventOut])
async def list_events(
    start: datetime | None = None,
    end: datetime | None = None,
    event_type: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    query = select(CalendarEvent).where(
        CalendarEvent.user_id == user_id,
        CalendarEvent.deleted_at.is_(None),
    )
    if start:
        query = query.where(CalendarEvent.start_time >= start)
    if end:
        query = query.where(CalendarEvent.start_time <= end)
    if event_type:
        query = query.where(CalendarEvent.event_type == event_type)
    query = query.order_by(CalendarEvent.start_time.asc().nullslast())
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/events/", response_model=CalendarEventOut, status_code=201)
async def create_event(body: CalendarEventCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    event = CalendarEvent(
        id=str(uuid4()),
        user_id=user_id,
        title=body.title,
        description=body.description,
        event_type=body.event_type,
        start_time=body.start_time,
        end_time=body.end_time,
        all_day=body.all_day,
        location=body.location,
        meeting_link=body.meeting_link,
        linked_meeting_id=body.linked_meeting_id,
        recurrence_rule=body.recurrence_rule,
        reminders=body.reminders,
        color=body.color,
    )
    db.add(event)
    await db.commit()
    await db.refresh(event)
    return event


@router.get("/events/{event_id}", response_model=CalendarEventOut)
async def get_event(event_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    event = await db.get(CalendarEvent, event_id)
    if not event or event.user_id != user_id or event.deleted_at:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.patch("/events/{event_id}", response_model=CalendarEventOut)
async def update_event(event_id: str, body: CalendarEventUpdate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    event = await db.get(CalendarEvent, event_id)
    if not event or event.user_id != user_id or event.deleted_at:
        raise HTTPException(status_code=404, detail="Event not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(event, field, value)
    event.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(event)
    return event


@router.delete("/events/{event_id}", status_code=204)
async def delete_event(event_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    event = await db.get(CalendarEvent, event_id)
    if not event or event.user_id != user_id or event.deleted_at:
        raise HTTPException(status_code=404, detail="Event not found")
    event.deleted_at = datetime.now(timezone.utc)
    await db.commit()


@router.get("/today/", response_model=list[CalendarEventOut])
async def today_events(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    result = await db.execute(
        select(CalendarEvent).where(
            CalendarEvent.user_id == user_id,
            CalendarEvent.deleted_at.is_(None),
            CalendarEvent.start_time >= start_of_day,
            CalendarEvent.start_time < end_of_day,
        ).order_by(CalendarEvent.start_time.asc())
    )
    return result.scalars().all()


@router.get("/week/", response_model=list[CalendarEventOut])
async def week_events(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = start_of_day + timedelta(days=7)
    result = await db.execute(
        select(CalendarEvent).where(
            CalendarEvent.user_id == user_id,
            CalendarEvent.deleted_at.is_(None),
            CalendarEvent.start_time >= start_of_day,
            CalendarEvent.start_time < end_of_week,
        ).order_by(CalendarEvent.start_time.asc())
    )
    return result.scalars().all()
