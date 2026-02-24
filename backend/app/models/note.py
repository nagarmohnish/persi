from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.user import Base


class Note(Base):
    __tablename__ = "notes"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=True)
    content_html = Column(Text, nullable=True)
    note_type = Column(String, default="idea")  # meeting_note, idea, journal, research, brainstorm, decision_log, retrospective
    linked_meeting_id = Column(String, ForeignKey("meetings.id"), nullable=True)
    linked_conversation_id = Column(String, ForeignKey("conversations.id"), nullable=True)
    tags = Column(JSONB, default=list)
    is_pinned = Column(Boolean, default=False)
    startup_stage = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="notes")
    versions = relationship("NoteVersion", back_populates="note", cascade="all, delete-orphan", order_by="NoteVersion.version_number.desc()")
    source_links = relationship("NoteLink", foreign_keys="NoteLink.source_note_id", back_populates="source_note", cascade="all, delete-orphan")
    target_links = relationship("NoteLink", foreign_keys="NoteLink.target_note_id", back_populates="target_note", cascade="all, delete-orphan")


class NoteVersion(Base):
    __tablename__ = "note_versions"

    id = Column(String, primary_key=True)
    note_id = Column(String, ForeignKey("notes.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=True)
    content_html = Column(Text, nullable=True)
    change_summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    note = relationship("Note", back_populates="versions")


class NoteLink(Base):
    __tablename__ = "note_links"

    id = Column(String, primary_key=True)
    source_note_id = Column(String, ForeignKey("notes.id"), nullable=False, index=True)
    target_note_id = Column(String, ForeignKey("notes.id"), nullable=False, index=True)
    link_type = Column(String, default="related")  # references, contradicts, builds_on, supersedes, related
    ai_generated = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    source_note = relationship("Note", foreign_keys=[source_note_id], back_populates="source_links")
    target_note = relationship("Note", foreign_keys=[target_note_id], back_populates="target_links")
