from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_db
from app.config import get_settings
from app.models.conversation import Conversation, Message
from app.schemas.conversation import ConversationCreate, ConversationUpdate, ConversationOut
from app.schemas.message import MessageCreate, MessageOut
from app.core.ai_engine import chat_completion
from app.core.stage_context import build_ai_context, build_system_prompt

router = APIRouter()


def _get_dev_user_id() -> str:
    """Return hardcoded dev user ID. Replaced by JWT auth in Phase 2."""
    return get_settings().dev_user_id


@router.get("/", response_model=list[ConversationOut])
async def list_conversations(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return result.scalars().all()


@router.post("/", response_model=ConversationOut, status_code=201)
async def create_conversation(
    body: ConversationCreate,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    conv = Conversation(
        id=str(uuid4()),
        user_id=user_id,
        title=body.title or "New conversation",
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if body.title is not None:
        conv.title = body.title
    conv.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(conv)
    return conv


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await db.delete(conv)
    await db.commit()


# --- Messages ---


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    return result.scalars().all()


@router.post("/{conversation_id}/messages", response_model=MessageOut, status_code=201)
async def create_message(
    conversation_id: str,
    body: MessageCreate,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg = Message(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role=body.role,
        content=body.content,
    )
    db.add(msg)
    conv.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(msg)
    return msg


@router.post("/{conversation_id}/ai-reply", response_model=MessageOut, status_code=201)
async def generate_ai_reply(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Generate an AI response based on conversation history and user context."""
    user_id = _get_dev_user_id()
    conv = await db.get(Conversation, conversation_id)
    if not conv or conv.user_id != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation messages for context
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()

    # Build AI context
    context = await build_ai_context(user_id, db)
    system_prompt = build_system_prompt(context)

    # Convert messages to Claude format (only user and assistant roles)
    chat_messages = []
    for m in messages:
        if m.role in ("user", "assistant"):
            chat_messages.append({"role": m.role, "content": m.content})

    # Generate AI response
    ai_response = await chat_completion(
        system_prompt=system_prompt,
        messages=chat_messages,
    )

    # Save assistant message
    assistant_msg = Message(
        id=str(uuid4()),
        conversation_id=conversation_id,
        role="assistant",
        content=ai_response,
    )
    db.add(assistant_msg)
    conv.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(assistant_msg)
    return assistant_msg
