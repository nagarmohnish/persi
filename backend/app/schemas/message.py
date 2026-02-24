from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime


class MessageCreate(BaseModel):
    role: Literal["user", "assistant", "system", "tool"] = "user"
    content: str


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}
