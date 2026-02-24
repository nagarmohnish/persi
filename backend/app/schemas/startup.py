from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class StartupCreate(BaseModel):
    name: str
    one_liner: Optional[str] = None
    problem_statement: Optional[str] = None
    target_audience: Optional[str] = None
    stage: str = "idea"
    industry: Optional[str] = None
    business_model: Optional[str] = None


class StartupUpdate(BaseModel):
    name: Optional[str] = None
    one_liner: Optional[str] = None
    problem_statement: Optional[str] = None
    target_audience: Optional[str] = None
    stage: Optional[str] = None
    industry: Optional[str] = None
    business_model: Optional[str] = None
    context_notes: Optional[str] = None


class StartupOut(BaseModel):
    id: str
    user_id: str
    name: Optional[str] = None
    one_liner: Optional[str] = None
    problem_statement: Optional[str] = None
    target_audience: Optional[str] = None
    stage: str
    industry: Optional[str] = None
    business_model: Optional[str] = None
    context_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
