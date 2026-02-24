from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_db
from app.config import get_settings
from app.models.travel import Trip, TripItem, TravelChecklist
from app.schemas.travel import TripCreate, TripUpdate, TripOut, TripItemCreate, TripItemOut, TravelChecklistOut

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


# --- Trips ---

@router.get("/", response_model=list[TripOut])
async def list_trips(status: str | None = None, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    query = select(Trip).where(Trip.user_id == user_id, Trip.deleted_at.is_(None))
    if status:
        query = query.where(Trip.status == status)
    query = query.order_by(Trip.start_date.desc().nullslast())
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/", response_model=TripOut, status_code=201)
async def create_trip(body: TripCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    trip = Trip(
        id=str(uuid4()),
        user_id=user_id,
        title=body.title,
        purpose=body.purpose,
        destination_city=body.destination_city,
        destination_country=body.destination_country,
        start_date=body.start_date,
        end_date=body.end_date,
        budget_cents=body.budget_cents,
        currency=body.currency,
        notes=body.notes,
    )
    db.add(trip)
    await db.commit()
    await db.refresh(trip)
    return trip


@router.get("/{trip_id}", response_model=TripOut)
async def get_trip(trip_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    trip = await db.get(Trip, trip_id)
    if not trip or trip.user_id != user_id or trip.deleted_at:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


@router.patch("/{trip_id}", response_model=TripOut)
async def update_trip(trip_id: str, body: TripUpdate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    trip = await db.get(Trip, trip_id)
    if not trip or trip.user_id != user_id or trip.deleted_at:
        raise HTTPException(status_code=404, detail="Trip not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(trip, field, value)
    trip.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(trip)
    return trip


@router.delete("/{trip_id}", status_code=204)
async def delete_trip(trip_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    trip = await db.get(Trip, trip_id)
    if not trip or trip.user_id != user_id or trip.deleted_at:
        raise HTTPException(status_code=404, detail="Trip not found")
    trip.deleted_at = datetime.now(timezone.utc)
    await db.commit()


# --- Trip Items ---

@router.get("/{trip_id}/items/", response_model=list[TripItemOut])
async def list_trip_items(trip_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    trip = await db.get(Trip, trip_id)
    if not trip or trip.user_id != user_id or trip.deleted_at:
        raise HTTPException(status_code=404, detail="Trip not found")
    result = await db.execute(
        select(TripItem).where(TripItem.trip_id == trip_id).order_by(TripItem.sort_order)
    )
    return result.scalars().all()


@router.post("/{trip_id}/items/", response_model=TripItemOut, status_code=201)
async def create_trip_item(trip_id: str, body: TripItemCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    trip = await db.get(Trip, trip_id)
    if not trip or trip.user_id != user_id or trip.deleted_at:
        raise HTTPException(status_code=404, detail="Trip not found")
    item = TripItem(
        id=str(uuid4()),
        trip_id=trip_id,
        item_type=body.item_type,
        title=body.title,
        description=body.description,
        start_time=body.start_time,
        end_time=body.end_time,
        location=body.location,
        address=body.address,
        confirmation_number=body.confirmation_number,
        cost_cents=body.cost_cents,
        currency=body.currency,
        booking_url=body.booking_url,
        status=body.status,
        sort_order=body.sort_order,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


@router.delete("/{trip_id}/items/{item_id}", status_code=204)
async def delete_trip_item(trip_id: str, item_id: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    trip = await db.get(Trip, trip_id)
    if not trip or trip.user_id != user_id or trip.deleted_at:
        raise HTTPException(status_code=404, detail="Trip not found")
    item = await db.get(TripItem, item_id)
    if not item or item.trip_id != trip_id:
        raise HTTPException(status_code=404, detail="Item not found")
    await db.delete(item)
    await db.commit()
