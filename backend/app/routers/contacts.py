from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_db
from app.config import get_settings
from app.models.contact import Contact, ContactInteraction
from app.schemas.contact import (
    ContactCreate, ContactUpdate, ContactOut,
    ContactInteractionCreate, ContactInteractionOut,
)

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


@router.get("/", response_model=list[ContactOut])
async def list_contacts(
    contact_type: str | None = None,
    company: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    query = select(Contact).where(Contact.user_id == user_id, Contact.deleted_at.is_(None))
    if contact_type:
        query = query.where(Contact.contact_type == contact_type)
    if company:
        query = query.where(Contact.company.ilike(f"%{company}%"))
    query = query.order_by(Contact.name.asc())
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/", response_model=ContactOut, status_code=201)
async def create_contact(body: ContactCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    contact = Contact(
        id=str(uuid4()),
        user_id=user_id,
        name=body.name,
        email=body.email,
        phone=body.phone,
        contact_type=body.contact_type,
        company=body.company,
        title=body.title,
        linkedin_url=body.linkedin_url,
        twitter_url=body.twitter_url,
        relationship_strength=body.relationship_strength,
        tags=body.tags,
        notes=body.notes,
    )
    db.add(contact)
    await db.commit()
    await db.refresh(contact)
    return contact


@router.get("/{contact_id}", response_model=ContactOut)
async def get_contact(contact_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    contact = await db.get(Contact, contact_id)
    if not contact or contact.user_id != user_id or contact.deleted_at:
        raise HTTPException(status_code=404, detail="Contact not found")
    return contact


@router.patch("/{contact_id}", response_model=ContactOut)
async def update_contact(contact_id: str, body: ContactUpdate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    contact = await db.get(Contact, contact_id)
    if not contact or contact.user_id != user_id or contact.deleted_at:
        raise HTTPException(status_code=404, detail="Contact not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(contact, field, value)
    contact.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(contact)
    return contact


@router.delete("/{contact_id}", status_code=204)
async def delete_contact(contact_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    contact = await db.get(Contact, contact_id)
    if not contact or contact.user_id != user_id or contact.deleted_at:
        raise HTTPException(status_code=404, detail="Contact not found")
    contact.deleted_at = datetime.now(timezone.utc)
    await db.commit()


# --- Interactions ---

@router.get("/{contact_id}/interactions/", response_model=list[ContactInteractionOut])
async def list_interactions(contact_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    contact = await db.get(Contact, contact_id)
    if not contact or contact.user_id != user_id or contact.deleted_at:
        raise HTTPException(status_code=404, detail="Contact not found")
    result = await db.execute(
        select(ContactInteraction)
        .where(ContactInteraction.contact_id == contact_id)
        .order_by(ContactInteraction.occurred_at.desc())
    )
    return result.scalars().all()


@router.post("/{contact_id}/interactions/", response_model=ContactInteractionOut, status_code=201)
async def log_interaction(contact_id: str, body: ContactInteractionCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    contact = await db.get(Contact, contact_id)
    if not contact or contact.user_id != user_id or contact.deleted_at:
        raise HTTPException(status_code=404, detail="Contact not found")

    interaction = ContactInteraction(
        id=str(uuid4()),
        contact_id=contact_id,
        user_id=user_id,
        interaction_type=body.interaction_type,
        summary=body.summary,
        linked_meeting_id=body.linked_meeting_id,
        linked_note_id=body.linked_note_id,
        occurred_at=body.occurred_at or datetime.now(timezone.utc),
    )
    db.add(interaction)

    # Update last_contacted_at
    contact.last_contacted_at = interaction.occurred_at
    contact.updated_at = datetime.now(timezone.utc)

    await db.commit()
    await db.refresh(interaction)
    return interaction


@router.get("/follow-ups/", response_model=list[ContactOut])
async def get_follow_ups(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(Contact).where(
            Contact.user_id == user_id,
            Contact.deleted_at.is_(None),
            Contact.next_follow_up_at.isnot(None),
            Contact.next_follow_up_at <= now,
        ).order_by(Contact.next_follow_up_at.asc())
    )
    return result.scalars().all()
