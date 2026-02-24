from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date


class JourneyStageOut(BaseModel):
    id: str
    user_id: str
    startup_id: str
    current_stage: str
    stage_started_at: datetime
    stage_completed_at: Optional[datetime] = None
    health_score: Optional[int] = None
    blockers: list = []
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JourneyMilestoneCreate(BaseModel):
    stage: str
    title: str
    description: Optional[str] = None
    category: str = "product"
    target_date: Optional[date] = None
    sort_order: int = 0


class JourneyMilestoneUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    target_date: Optional[date] = None
    completed_date: Optional[date] = None
    sort_order: Optional[int] = None


class JourneyMilestoneOut(BaseModel):
    id: str
    user_id: str
    stage: str
    title: str
    description: Optional[str] = None
    category: str
    status: str
    target_date: Optional[date] = None
    completed_date: Optional[date] = None
    evidence: list = []
    ai_suggestions: list = []
    sort_order: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class JourneyPlaybookOut(BaseModel):
    id: str
    stage: str
    category: Optional[str] = None
    title: str
    description: Optional[str] = None
    tasks: list = []
    frameworks: list = []
    common_mistakes: list = []
    success_criteria: list = []

    model_config = {"from_attributes": True}
