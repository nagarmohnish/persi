from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.config import get_settings
from app.core.ai_engine import chat_completion, generate_summary
from app.core.stage_context import build_ai_context, build_system_prompt
from app.models.note import Note
from app.models.meeting import Meeting
from app.models.contact import Contact
from app.models.travel import Trip

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


class AskRequest(BaseModel):
    query: str
    conversation_history: list[dict] = []


class AskResponse(BaseModel):
    response: str
    stage: str


class EnhanceNoteRequest(BaseModel):
    content: str
    action: str = "summarize"  # summarize, expand, improve


class EnhanceNoteResponse(BaseModel):
    enhanced_content: str


class TripSuggestionsRequest(BaseModel):
    destination_city: str
    destination_country: str
    purpose: str
    duration_days: Optional[int] = None


class TripSuggestionsResponse(BaseModel):
    suggestions: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    id: str
    type: str
    title: str
    subtitle: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


@router.post("/ask", response_model=AskResponse)
async def ask_ai(body: AskRequest, db: AsyncSession = Depends(get_db)):
    """General AI query with full context injection."""
    user_id = _get_dev_user_id()
    context = await build_ai_context(user_id, db)
    system_prompt = build_system_prompt(context)

    messages = list(body.conversation_history)
    messages.append({"role": "user", "content": body.query})

    response_text = await chat_completion(
        system_prompt=system_prompt,
        messages=messages,
    )

    return AskResponse(
        response=response_text,
        stage=context["current_stage"],
    )


@router.post("/enhance-note", response_model=EnhanceNoteResponse)
async def enhance_note(body: EnhanceNoteRequest):
    """Enhance a note using AI (summarize, expand, or improve)."""
    action_prompts = {
        "summarize": "Create a concise summary of the following note, highlighting key insights and action items:\n\n",
        "expand": "Expand on the following note with more detail, examples, and related ideas. Keep the same tone and structure:\n\n",
        "improve": "Improve the following note for clarity, structure, and completeness. Fix any grammar issues and enhance readability:\n\n",
    }

    prompt = action_prompts.get(body.action, action_prompts["summarize"])
    result = await generate_summary(prompt + body.content, "note")

    return EnhanceNoteResponse(enhanced_content=result)


@router.post("/trip-suggestions", response_model=TripSuggestionsResponse)
async def trip_suggestions(body: TripSuggestionsRequest):
    """Get AI-powered travel suggestions for a trip."""
    duration_text = f" for {body.duration_days} days" if body.duration_days else ""
    prompt = (
        f"I'm planning a {body.purpose} trip to {body.destination_city}, "
        f"{body.destination_country}{duration_text}. "
        f"Please suggest: 1) Key places to visit or meetings to arrange, "
        f"2) Practical tips for the destination, "
        f"3) Packing suggestions, "
        f"4) Useful apps or services for the area. "
        f"Keep it concise and practical for a startup founder."
    )

    result = await generate_summary(prompt, "general")
    return TripSuggestionsResponse(suggestions=result)


@router.post("/search", response_model=SearchResponse)
async def global_search(body: SearchRequest, db: AsyncSession = Depends(get_db)):
    """Search across all entities (notes, meetings, contacts, trips)."""
    user_id = _get_dev_user_id()
    query = f"%{body.query}%"
    results: list[SearchResult] = []

    # Search notes
    notes_q = await db.execute(
        select(Note)
        .where(Note.user_id == user_id, Note.deleted_at.is_(None))
        .where(or_(Note.title.ilike(query), Note.content.ilike(query)))
        .limit(body.limit)
    )
    for note in notes_q.scalars():
        results.append(SearchResult(
            id=note.id, type="note", title=note.title,
            subtitle=note.note_type,
        ))

    # Search meetings
    meetings_q = await db.execute(
        select(Meeting)
        .where(Meeting.user_id == user_id, Meeting.deleted_at.is_(None))
        .where(or_(Meeting.title.ilike(query), Meeting.description.ilike(query)))
        .limit(body.limit)
    )
    for meeting in meetings_q.scalars():
        results.append(SearchResult(
            id=meeting.id, type="meeting", title=meeting.title,
            subtitle=meeting.meeting_type,
        ))

    # Search contacts
    contacts_q = await db.execute(
        select(Contact)
        .where(Contact.user_id == user_id, Contact.deleted_at.is_(None))
        .where(or_(Contact.name.ilike(query), Contact.email.ilike(query), Contact.company.ilike(query)))
        .limit(body.limit)
    )
    for contact in contacts_q.scalars():
        results.append(SearchResult(
            id=contact.id, type="contact", title=contact.name,
            subtitle=contact.company,
        ))

    # Search trips
    trips_q = await db.execute(
        select(Trip)
        .where(Trip.user_id == user_id, Trip.deleted_at.is_(None))
        .where(or_(Trip.title.ilike(query), Trip.destination_city.ilike(query)))
        .limit(body.limit)
    )
    for trip in trips_q.scalars():
        results.append(SearchResult(
            id=trip.id, type="trip", title=trip.title,
            subtitle=trip.destination_city,
        ))

    return SearchResponse(results=results[:body.limit])
