from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import uuid4
from datetime import datetime, timezone

from app.database import get_db
from app.config import get_settings
from app.models.journey import JourneyStage, JourneyMilestone, JourneyPlaybook
from app.schemas.journey import (
    JourneyStageOut, JourneyMilestoneCreate, JourneyMilestoneUpdate,
    JourneyMilestoneOut, JourneyPlaybookOut,
)

router = APIRouter()


def _get_dev_user_id() -> str:
    return get_settings().dev_user_id


# --- Stage ---

@router.get("/current/", response_model=JourneyStageOut | None)
async def get_current_stage(db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    result = await db.execute(
        select(JourneyStage)
        .where(JourneyStage.user_id == user_id, JourneyStage.stage_completed_at.is_(None))
        .order_by(JourneyStage.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


@router.post("/stage/", response_model=JourneyStageOut, status_code=201)
async def set_stage(startup_id: str, stage: str, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    # Complete current stage if exists
    result = await db.execute(
        select(JourneyStage).where(
            JourneyStage.user_id == user_id,
            JourneyStage.stage_completed_at.is_(None),
        )
    )
    current = result.scalar_one_or_none()
    if current:
        current.stage_completed_at = datetime.now(timezone.utc)

    new_stage = JourneyStage(
        id=str(uuid4()),
        user_id=user_id,
        startup_id=startup_id,
        current_stage=stage,
    )
    db.add(new_stage)
    await db.commit()
    await db.refresh(new_stage)
    return new_stage


# --- Milestones ---

@router.get("/milestones/", response_model=list[JourneyMilestoneOut])
async def list_milestones(stage: str | None = None, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    query = select(JourneyMilestone).where(JourneyMilestone.user_id == user_id)
    if stage:
        query = query.where(JourneyMilestone.stage == stage)
    query = query.order_by(JourneyMilestone.sort_order)
    result = await db.execute(query)
    return result.scalars().all()


@router.post("/milestones/", response_model=JourneyMilestoneOut, status_code=201)
async def create_milestone(body: JourneyMilestoneCreate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    milestone = JourneyMilestone(
        id=str(uuid4()),
        user_id=user_id,
        stage=body.stage,
        title=body.title,
        description=body.description,
        category=body.category,
        target_date=body.target_date,
        sort_order=body.sort_order,
    )
    db.add(milestone)
    await db.commit()
    await db.refresh(milestone)
    return milestone


@router.patch("/milestones/{milestone_id}", response_model=JourneyMilestoneOut)
async def update_milestone(milestone_id: str, body: JourneyMilestoneUpdate, db: AsyncSession = Depends(get_db)):
    user_id = _get_dev_user_id()
    milestone = await db.get(JourneyMilestone, milestone_id)
    if not milestone or milestone.user_id != user_id:
        raise HTTPException(status_code=404, detail="Milestone not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(milestone, field, value)
    milestone.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(milestone)
    return milestone


# --- Playbooks ---

@router.get("/playbooks/", response_model=list[JourneyPlaybookOut])
async def list_playbooks(stage: str | None = None, db: AsyncSession = Depends(get_db)):
    query = select(JourneyPlaybook)
    if stage:
        query = query.where(JourneyPlaybook.stage == stage)
    result = await db.execute(query)
    return result.scalars().all()


@router.get("/playbooks/{playbook_id}", response_model=JourneyPlaybookOut)
async def get_playbook(playbook_id: str, db: AsyncSession = Depends(get_db)):
    playbook = await db.get(JourneyPlaybook, playbook_id)
    if not playbook:
        raise HTTPException(status_code=404, detail="Playbook not found")
    return playbook
