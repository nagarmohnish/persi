from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date


class TripCreate(BaseModel):
    title: str
    purpose: str = "other"
    destination_city: Optional[str] = None
    destination_country: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    budget_cents: Optional[int] = None
    currency: str = "USD"
    notes: Optional[str] = None


class TripUpdate(BaseModel):
    title: Optional[str] = None
    purpose: Optional[str] = None
    destination_city: Optional[str] = None
    destination_country: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Optional[str] = None
    budget_cents: Optional[int] = None
    currency: Optional[str] = None
    notes: Optional[str] = None


class TripOut(BaseModel):
    id: str
    user_id: str
    title: str
    purpose: str
    destination_city: Optional[str] = None
    destination_country: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: str
    budget_cents: Optional[int] = None
    currency: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TripItemCreate(BaseModel):
    item_type: str = "other"
    title: str
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    address: Optional[str] = None
    confirmation_number: Optional[str] = None
    cost_cents: Optional[int] = None
    currency: str = "USD"
    booking_url: Optional[str] = None
    status: str = "researching"
    sort_order: int = 0


class TripItemOut(BaseModel):
    id: str
    trip_id: str
    item_type: str
    title: str
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    address: Optional[str] = None
    confirmation_number: Optional[str] = None
    cost_cents: Optional[int] = None
    currency: str
    booking_url: Optional[str] = None
    status: str
    sort_order: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TravelChecklistOut(BaseModel):
    id: str
    trip_id: str
    title: str
    items: list = []
    ai_generated: bool
    created_at: datetime

    model_config = {"from_attributes": True}
