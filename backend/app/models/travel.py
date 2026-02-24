from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Boolean, Integer, BigInteger, Date
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from app.models.user import Base


class Trip(Base):
    __tablename__ = "trips"

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    purpose = Column(String, default="other")  # investor_meeting, conference, team_offsite, customer_visit, personal, other
    destination_city = Column(String, nullable=True)
    destination_country = Column(String, nullable=True)
    start_date = Column(Date, nullable=True)
    end_date = Column(Date, nullable=True)
    status = Column(String, default="planning")  # planning, booked, in_progress, completed, cancelled
    budget_cents = Column(BigInteger, nullable=True)
    currency = Column(String, default="USD")
    linked_meeting_ids = Column(JSONB, default=list)
    notes = Column(Text, nullable=True)
    startup_stage = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="trips")
    items = relationship("TripItem", back_populates="trip", cascade="all, delete-orphan", order_by="TripItem.sort_order")
    checklists = relationship("TravelChecklist", back_populates="trip", cascade="all, delete-orphan")


class TripItem(Base):
    __tablename__ = "trip_items"

    id = Column(String, primary_key=True)
    trip_id = Column(String, ForeignKey("trips.id"), nullable=False, index=True)
    item_type = Column(String, default="other")  # flight, hotel, car_rental, restaurant, activity, meeting, transport, other
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    location = Column(Text, nullable=True)
    address = Column(Text, nullable=True)
    confirmation_number = Column(String, nullable=True)
    cost_cents = Column(BigInteger, nullable=True)
    currency = Column(String, default="USD")
    booking_url = Column(Text, nullable=True)
    status = Column(String, default="researching")  # researching, booked, confirmed, cancelled
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    trip = relationship("Trip", back_populates="items")


class TravelChecklist(Base):
    __tablename__ = "travel_checklists"

    id = Column(String, primary_key=True)
    trip_id = Column(String, ForeignKey("trips.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    items = Column(JSONB, default=list)  # [{item, checked, category}]
    ai_generated = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    trip = relationship("Trip", back_populates="checklists")
