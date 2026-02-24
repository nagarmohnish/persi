from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.user import Base


class Startup(Base):
    __tablename__ = "startups"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=True)
    one_liner = Column(String, nullable=True)
    problem_statement = Column(Text, nullable=True)
    target_audience = Column(String, nullable=True)
    stage = Column(
        String,
        default="idea",
    )  # idea, validation, mvp, launched, fundraising, scaling
    industry = Column(String, nullable=True)
    business_model = Column(String, nullable=True)
    founding_team = Column(JSONB, default=list)  # [{name, role, background}]
    context_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    user = relationship("User", back_populates="startups")
