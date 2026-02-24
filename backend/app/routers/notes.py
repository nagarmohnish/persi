from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_db
from app.config import get_settings
from app.models.note import Note, NoteVersion
from app.schemas.note import NoteCreate, NoteUpdate, NoteOut, NoteVersionOut

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


@router.get("/", response_model=list[NoteOut])
async def list_notes(
    note_type: str | None = None,
    is_pinned: bool | None = None,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    query = select(Note).where(Note.user_id == user_id, Note.deleted_at.is_(None))
    if note_type:
        query = query.where(Note.note_type == note_type)
    if is_pinned is not None:
        query = query.where(Note.is_pinned == is_pinned)
    query = query.order_by(Note.updated_at.desc())
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/", response_model=NoteOut, status_code=201)
async def create_note(body: NoteCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    note = Note(
        id=str(uuid4()),
        user_id=user_id,
        title=body.title,
        content=body.content,
        content_html=body.content_html,
        note_type=body.note_type,
        linked_meeting_id=body.linked_meeting_id,
        linked_conversation_id=body.linked_conversation_id,
        tags=body.tags,
        startup_stage=body.startup_stage,
    )
    db.add(note)
    await db.commit()
    await db.refresh(note)
    return note


@router.get("/{note_id}", response_model=NoteOut)
async def get_note(note_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    note = await db.get(Note, note_id)
    if not note or note.user_id != user_id or note.deleted_at:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.put("/{note_id}", response_model=NoteOut)
async def update_note(note_id: str, body: NoteUpdate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    note = await db.get(Note, note_id)
    if not note or note.user_id != user_id or note.deleted_at:
        raise HTTPException(status_code=404, detail="Note not found")

    # Save version before updating
    version_count = (await db.execute(
        select(NoteVersion).where(NoteVersion.note_id == note_id)
    )).scalars().all()
    version = NoteVersion(
        id=str(uuid4()),
        note_id=note_id,
        version_number=len(version_count) + 1,
        content=note.content,
        content_html=note.content_html,
    )
    db.add(version)

    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(note, field, value)
    note.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(note)
    return note


@router.delete("/{note_id}", status_code=204)
async def delete_note(note_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    note = await db.get(Note, note_id)
    if not note or note.user_id != user_id or note.deleted_at:
        raise HTTPException(status_code=404, detail="Note not found")
    note.deleted_at = datetime.now(timezone.utc)
    await db.commit()


@router.post("/{note_id}/pin", response_model=NoteOut)
async def toggle_pin(note_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    note = await db.get(Note, note_id)
    if not note or note.user_id != user_id or note.deleted_at:
        raise HTTPException(status_code=404, detail="Note not found")
    note.is_pinned = not note.is_pinned
    await db.commit()
    await db.refresh(note)
    return note


@router.get("/{note_id}/versions", response_model=list[NoteVersionOut])
async def list_versions(note_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    note = await db.get(Note, note_id)
    if not note or note.user_id != user_id or note.deleted_at:
        raise HTTPException(status_code=404, detail="Note not found")
    result = await db.execute(
        select(NoteVersion).where(NoteVersion.note_id == note_id).order_by(NoteVersion.version_number.desc())
    )
    return result.scalars().all()
