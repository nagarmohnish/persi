from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_db
from app.config import get_settings
from app.models.meeting import Meeting, MeetingParticipant, MeetingTemplate, MeetingSummary
from app.schemas.meeting import (
    MeetingCreate, MeetingUpdate, MeetingOut,
    MeetingParticipantCreate, MeetingParticipantOut,
    MeetingTemplateCreate, MeetingTemplateOut,
    MeetingSummaryOut,
)
from app.core.ai_engine import generate_summary

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


@router.get("/", response_model=list[MeetingOut])
async def list_meetings(
    status: str | None = None,
    meeting_type: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    query = select(Meeting).where(Meeting.user_id == user_id, Meeting.deleted_at.is_(None))
    if status:
        query = query.where(Meeting.status == status)
    if meeting_type:
        query = query.where(Meeting.meeting_type == meeting_type)
    query = query.order_by(Meeting.start_time.desc().nullslast())
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/", response_model=MeetingOut, status_code=201)
async def create_meeting(body: MeetingCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = Meeting(
        id=str(uuid4()),
        user_id=user_id,
        title=body.title,
        description=body.description,
        meeting_type=body.meeting_type,
        start_time=body.start_time,
        end_time=body.end_time,
        duration_minutes=body.duration_minutes,
        location=body.location,
        meeting_link=body.meeting_link,
        platform=body.platform,
        is_recurring=body.is_recurring,
        recurrence_rule=body.recurrence_rule,
    )
    db.add(meeting)
    await db.commit()
    await db.refresh(meeting)
    return meeting


@router.get("/{meeting_id}", response_model=MeetingOut)
async def get_meeting(meeting_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.patch("/{meeting_id}", response_model=MeetingOut)
async def update_meeting(meeting_id: str, body: MeetingUpdate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(meeting, field, value)
    meeting.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(meeting)
    return meeting


@router.delete("/{meeting_id}", status_code=204)
async def delete_meeting(meeting_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")
    meeting.deleted_at = datetime.now(timezone.utc)
    await db.commit()


# --- Participants ---

@router.get("/{meeting_id}/participants", response_model=list[MeetingParticipantOut])
async def list_participants(meeting_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")
    result = await db.execute(
        select(MeetingParticipant).where(MeetingParticipant.meeting_id == meeting_id)
    )
    return result.scalars().all()


@router.post("/{meeting_id}/participants", response_model=MeetingParticipantOut, status_code=201)
async def add_participant(meeting_id: str, body: MeetingParticipantCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")
    participant = MeetingParticipant(
        id=str(uuid4()),
        meeting_id=meeting_id,
        name=body.name,
        email=body.email,
        role=body.role,
    )
    db.add(participant)
    await db.commit()
    await db.refresh(participant)
    return participant


@router.delete("/{meeting_id}/participants/{participant_id}", status_code=204)
async def remove_participant(meeting_id: str, participant_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")
    participant = await db.get(MeetingParticipant, participant_id)
    if not participant or participant.meeting_id != meeting_id:
        raise HTTPException(status_code=404, detail="Participant not found")
    await db.delete(participant)
    await db.commit()


# --- Templates ---

@router.get("/templates/", response_model=list[MeetingTemplateOut])
async def list_templates(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    result = await db.execute(
        select(MeetingTemplate).where(MeetingTemplate.user_id == user_id)
    )
    return result.scalars().all()


@router.post("/templates/", response_model=MeetingTemplateOut, status_code=201)
async def create_template(body: MeetingTemplateCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    template = MeetingTemplate(
        id=str(uuid4()),
        user_id=user_id,
        name=body.name,
        meeting_type=body.meeting_type,
        default_agenda=body.default_agenda,
        default_duration_minutes=body.default_duration_minutes,
        pre_meeting_prompts=body.pre_meeting_prompts,
        post_meeting_prompts=body.post_meeting_prompts,
    )
    db.add(template)
    await db.commit()
    await db.refresh(template)
    return template


# --- Summaries ---

@router.get("/{meeting_id}/summary", response_model=MeetingSummaryOut | None)
async def get_meeting_summary(meeting_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")
    result = await db.execute(
        select(MeetingSummary)
        .where(MeetingSummary.meeting_id == meeting_id)
        .order_by(MeetingSummary.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


@router.post("/{meeting_id}/summary", response_model=MeetingSummaryOut, status_code=201)
async def generate_meeting_summary(meeting_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    meeting = await db.get(Meeting, meeting_id)
    if not meeting or meeting.user_id != user_id or meeting.deleted_at:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Build context from meeting details
    participants_result = await db.execute(
        select(MeetingParticipant).where(MeetingParticipant.meeting_id == meeting_id)
    )
    participants = participants_result.scalars().all()
    participant_names = ", ".join(p.name for p in participants)

    text = f"Meeting: {meeting.title}\nType: {meeting.meeting_type}\n"
    if meeting.description:
        text += f"Agenda/Description: {meeting.description}\n"
    if participant_names:
        text += f"Participants: {participant_names}\n"

    summary_text = await generate_summary(text, summary_type="meeting")

    summary = MeetingSummary(
        id=str(uuid4()),
        meeting_id=meeting_id,
        summary=summary_text,
        generated_by="ai",
    )
    db.add(summary)
    await db.commit()
    await db.refresh(summary)
    return summary
