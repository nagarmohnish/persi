from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, Integer, Date
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.user import Base


class JourneyTask(Base):
    __tablename__ = "journey_tasks"

    id = Column(String, primary_key=True)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False)
    stage = Column(String, nullable=False)  # idea, validation, mvp, launched, fundraising, scaling
    task_title = Column(String, nullable=False)
    task_description = Column(Text, nullable=True)
    status = Column(String, default="pending")  # pending, in_progress, completed, skipped
    notes = Column(Text, nullable=True)
    ai_generated = Column(Boolean, default=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class JourneyStage(Base):
    __tablename__ = "journey_stages"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False, index=True)
    current_stage = Column(String, nullable=False)  # ideation, validation, mvp, launch, growth, fundraising, scaling
    stage_started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    stage_completed_at = Column(DateTime(timezone=True), nullable=True)
    health_score = Column(Integer, nullable=True)  # 0-100
    blockers = Column(JSONB, default=list)  # [{description, severity, suggested_action}]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="journey_stages")


class JourneyMilestone(Base):
    __tablename__ = "journey_milestones"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    stage = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String, default="product")  # product, customer, revenue, team, funding, legal, operations
    status = Column(String, default="not_started")  # not_started, in_progress, completed, skipped
    target_date = Column(Date, nullable=True)
    completed_date = Column(Date, nullable=True)
    evidence = Column(JSONB, default=list)  # [{type, description, link}]
    ai_suggestions = Column(JSONB, default=list)
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="journey_milestones")


class JourneyPlaybook(Base):
    __tablename__ = "journey_playbooks"

    id = Column(String, primary_key=True)
    stage = Column(String, nullable=False, index=True)
    category = Column(String, nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    tasks = Column(JSONB, default=list)  # [{task, description, resources, estimated_hours}]
    frameworks = Column(JSONB, default=list)  # [{name, description, when_to_use}]
    common_mistakes = Column(JSONB, default=list)
    success_criteria = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
