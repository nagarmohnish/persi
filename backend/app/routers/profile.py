from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4

from app.database import get_db
from app.config import get_settings
from app.models.user import User
from app.models.startup import Startup
from app.schemas.user import UserOut, UserUpdate
from app.schemas.startup import StartupCreate, StartupUpdate, StartupOut

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


@router.get("/me", response_model=UserOut)
async def get_profile(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.patch("/me", response_model=UserOut)
async def update_profile(body: UserUpdate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/startup", response_model=StartupOut | None)
async def get_startup(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    result = await db.execute(
        select(Startup).where(Startup.user_id == user_id).limit(1)
    )
    startup = result.scalar_one_or_none()
    return startup


@router.post("/startup", response_model=StartupOut, status_code=201)
async def create_startup(
    body: StartupCreate,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    startup = Startup(
        id=str(uuid4()),
        user_id=user_id,
        name=body.name,
        one_liner=body.one_liner,
        problem_statement=body.problem_statement,
        target_audience=body.target_audience,
        stage=body.stage,
        industry=body.industry,
        business_model=body.business_model,
    )
    db.add(startup)
    await db.commit()
    await db.refresh(startup)
    return startup


@router.put("/startup/{startup_id}", response_model=StartupOut)
async def update_startup(
    startup_id: str,
    body: StartupUpdate,
    db: AsyncSession = Depends(get_db),
):
    user_id = _get_dev_user_id()
    startup = await db.get(Startup, startup_id)
    if not startup or startup.user_id != user_id:
        raise HTTPException(status_code=404, detail="Startup not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(startup, field, value)
    await db.commit()
    await db.refresh(startup)
    return startup
