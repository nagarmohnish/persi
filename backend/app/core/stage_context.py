from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone, timedelta

from app.models.user import User
from app.models.startup import Startup
from app.models.meeting import Meeting
from app.models.note import Note
from app.models.journey import JourneyStage, JourneyMilestone
from app.models.contact import Contact
from app.models.calendar import CalendarEvent

STAGE_PROFILES = {
    "ideation": {
        "focus_areas": ["problem validation", "customer discovery", "market sizing", "competitive landscape"],
        "frameworks": ["Lean Canvas", "Jobs-to-be-Done", "TAM/SAM/SOM", "Problem-Solution Fit"],
        "tone": "exploratory, questioning, Socratic",
        "common_tasks": ["customer interview prep", "problem statement refinement", "initial market research"],
        "red_flags": ["building before validating", "solution-first thinking", "ignoring customer feedback"],
    },
    "idea": {  # alias
        "focus_areas": ["problem validation", "customer discovery", "market sizing", "competitive landscape"],
        "frameworks": ["Lean Canvas", "Jobs-to-be-Done", "TAM/SAM/SOM", "Problem-Solution Fit"],
        "tone": "exploratory, questioning, Socratic",
        "common_tasks": ["customer interview prep", "problem statement refinement", "initial market research"],
        "red_flags": ["building before validating", "solution-first thinking", "ignoring customer feedback"],
    },
    "validation": {
        "focus_areas": ["customer interviews", "prototype testing", "willingness to pay", "channel discovery"],
        "frameworks": ["Customer Development", "Experiment Board", "Pirate Metrics", "Value Proposition Canvas"],
        "tone": "rigorous, evidence-based, challenging assumptions",
        "common_tasks": ["interview synthesis", "experiment design", "landing page creation"],
        "red_flags": ["confirmation bias", "vanity metrics", "premature scaling"],
    },
    "mvp": {
        "focus_areas": ["core feature set", "technical architecture", "user feedback loops", "iteration speed"],
        "frameworks": ["MoSCoW prioritization", "Story Mapping", "Build-Measure-Learn", "Kano Model"],
        "tone": "focused, prioritization-driven, shipping-oriented",
        "common_tasks": ["feature prioritization", "sprint planning", "user testing", "bug triage"],
        "red_flags": ["feature creep", "perfectionism", "no user feedback", "over-engineering"],
    },
    "launch": {
        "focus_areas": ["go-to-market", "initial traction", "PR/marketing", "onboarding optimization"],
        "frameworks": ["Launch Checklist", "Growth Loops", "AARRR Funnel", "Positioning Statement"],
        "tone": "energetic, tactical, action-oriented",
        "common_tasks": ["launch plan", "press outreach", "social media strategy", "onboarding flows"],
        "red_flags": ["no launch plan", "poor onboarding", "ignoring churn signals"],
    },
    "launched": {  # alias
        "focus_areas": ["go-to-market", "initial traction", "PR/marketing", "onboarding optimization"],
        "frameworks": ["Launch Checklist", "Growth Loops", "AARRR Funnel", "Positioning Statement"],
        "tone": "energetic, tactical, action-oriented",
        "common_tasks": ["launch plan", "press outreach", "social media strategy", "onboarding flows"],
        "red_flags": ["no launch plan", "poor onboarding", "ignoring churn signals"],
    },
    "growth": {
        "focus_areas": ["retention", "unit economics", "scalable channels", "team building"],
        "frameworks": ["North Star Metric", "Growth Accounting", "Cohort Analysis", "OKRs"],
        "tone": "analytical, systems-thinking, metrics-driven",
        "common_tasks": ["growth experiment design", "funnel analysis", "hiring plans", "process documentation"],
        "red_flags": ["growing without retention", "unsustainable CAC", "founder bottleneck"],
    },
    "fundraising": {
        "focus_areas": ["pitch deck", "financial model", "investor targeting", "due diligence prep"],
        "frameworks": ["Pitch Deck Structure", "Cap Table Modeling", "Investor CRM", "SAFE/Priced Round"],
        "tone": "strategic, narrative-focused, numbers-backed",
        "common_tasks": ["pitch practice", "deck review", "investor email drafts", "term sheet analysis"],
        "red_flags": ["fundraising too early", "bad unit economics", "no traction story", "unclear use of funds"],
    },
    "scaling": {
        "focus_areas": ["systems & processes", "leadership team", "culture", "international expansion"],
        "frameworks": ["EOS/Traction", "Scaling Up", "OKRs at Scale", "Conway's Law"],
        "tone": "strategic, delegation-focused, long-term thinking",
        "common_tasks": ["org design", "process automation", "board reporting", "strategic planning"],
        "red_flags": ["founder doing everything", "no processes", "culture debt", "key person dependencies"],
    },
}


async def build_ai_context(user_id: str, db: AsyncSession) -> dict:
    """Build full context payload for AI calls."""
    user = await db.get(User, user_id)

    # Get startup
    result = await db.execute(
        select(Startup).where(Startup.user_id == user_id).limit(1)
    )
    startup = result.scalar_one_or_none()

    # Get current stage
    result = await db.execute(
        select(JourneyStage)
        .where(JourneyStage.user_id == user_id, JourneyStage.stage_completed_at.is_(None))
        .order_by(JourneyStage.created_at.desc())
        .limit(1)
    )
    journey_stage = result.scalar_one_or_none()
    current_stage = journey_stage.current_stage if journey_stage else (startup.stage if startup else "ideation")

    stage_profile = STAGE_PROFILES.get(current_stage, STAGE_PROFILES["ideation"])

    # Recent meetings (last 5)
    result = await db.execute(
        select(Meeting)
        .where(Meeting.user_id == user_id, Meeting.deleted_at.is_(None))
        .order_by(Meeting.start_time.desc().nullslast())
        .limit(5)
    )
    recent_meetings = result.scalars().all()

    # Upcoming meetings (next 5)
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(Meeting)
        .where(Meeting.user_id == user_id, Meeting.deleted_at.is_(None), Meeting.start_time >= now)
        .order_by(Meeting.start_time.asc())
        .limit(5)
    )
    upcoming_meetings = result.scalars().all()

    # Recent notes (last 10)
    result = await db.execute(
        select(Note)
        .where(Note.user_id == user_id, Note.deleted_at.is_(None))
        .order_by(Note.updated_at.desc())
        .limit(10)
    )
    recent_notes = result.scalars().all()

    # Active milestones
    result = await db.execute(
        select(JourneyMilestone)
        .where(JourneyMilestone.user_id == user_id, JourneyMilestone.status.in_(["not_started", "in_progress"]))
        .order_by(JourneyMilestone.sort_order)
        .limit(10)
    )
    active_milestones = result.scalars().all()

    # Key contacts
    result = await db.execute(
        select(Contact)
        .where(Contact.user_id == user_id, Contact.deleted_at.is_(None))
        .order_by(Contact.last_contacted_at.desc().nullslast())
        .limit(10)
    )
    key_contacts = result.scalars().all()

    # Today's events
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    result = await db.execute(
        select(CalendarEvent)
        .where(
            CalendarEvent.user_id == user_id,
            CalendarEvent.deleted_at.is_(None),
            CalendarEvent.start_time >= start_of_day,
            CalendarEvent.start_time < end_of_day,
        )
        .order_by(CalendarEvent.start_time.asc())
    )
    today_events = result.scalars().all()

    return {
        "user": {
            "name": user.full_name or "Founder",
            "email": user.email,
        } if user else {"name": "Founder", "email": ""},
        "startup": {
            "name": startup.name or "Unnamed startup",
            "one_liner": startup.one_liner or "",
            "problem_statement": startup.problem_statement or "",
            "target_audience": startup.target_audience or "",
            "stage": startup.stage,
            "industry": startup.industry or "",
            "business_model": startup.business_model or "",
            "founding_team": startup.founding_team or [],
        } if startup else None,
        "current_stage": current_stage,
        "stage_profile": stage_profile,
        "recent_meetings": [{"title": m.title, "type": m.meeting_type, "time": str(m.start_time)} for m in recent_meetings],
        "upcoming_meetings": [{"title": m.title, "type": m.meeting_type, "time": str(m.start_time)} for m in upcoming_meetings],
        "recent_notes": [{"title": n.title, "type": n.note_type} for n in recent_notes],
        "active_milestones": [{"title": m.title, "category": m.category, "status": m.status} for m in active_milestones],
        "key_contacts": [{"name": c.name, "type": c.contact_type, "company": c.company} for c in key_contacts],
        "today_schedule": [{"title": e.title, "type": e.event_type, "time": str(e.start_time)} for e in today_events],
        "blockers": journey_stage.blockers if journey_stage and journey_stage.blockers else [],
    }


def build_system_prompt(context: dict) -> str:
    """Build the stage-aware system prompt from context."""
    startup = context.get("startup")
    stage_profile = context.get("stage_profile", {})

    startup_section = ""
    if startup:
        startup_section = f"""
STARTUP CONTEXT:
- Name: {startup['name']}
- One-liner: {startup['one_liner']}
- Problem: {startup['problem_statement']}
- Target audience: {startup['target_audience']}
- Industry: {startup['industry']}
- Business model: {startup['business_model']}
- Team: {', '.join(f"{t.get('name', '')} ({t.get('role', '')})" for t in startup.get('founding_team', []))}
"""

    meetings_section = ""
    if context.get("upcoming_meetings"):
        meetings_section = "UPCOMING MEETINGS:\n" + "\n".join(
            f"- {m['title']} ({m['type']}) at {m['time']}" for m in context["upcoming_meetings"]
        ) + "\n"

    notes_section = ""
    if context.get("recent_notes"):
        notes_section = "RECENT NOTES:\n" + "\n".join(
            f"- {n['title']} ({n['type']})" for n in context["recent_notes"][:5]
        ) + "\n"

    milestones_section = ""
    if context.get("active_milestones"):
        milestones_section = "ACTIVE MILESTONES:\n" + "\n".join(
            f"- [{m['status']}] {m['title']} ({m['category']})" for m in context["active_milestones"]
        ) + "\n"

    schedule_section = ""
    if context.get("today_schedule"):
        schedule_section = "TODAY'S SCHEDULE:\n" + "\n".join(
            f"- {e['title']} ({e['type']}) at {e['time']}" for e in context["today_schedule"]
        ) + "\n"

    blockers_section = ""
    if context.get("blockers"):
        blockers_section = "ACTIVE BLOCKERS:\n" + "\n".join(
            f"- {b.get('description', 'Unknown')}" for b in context["blockers"]
        ) + "\n"

    return f"""You are Persi, an AI co-pilot for startup founders. You are currently assisting {context['user']['name']}.

CURRENT STAGE: {context['current_stage']}
At this stage, your primary focus areas are: {', '.join(stage_profile.get('focus_areas', []))}
Relevant frameworks: {', '.join(stage_profile.get('frameworks', []))}
Communication style: {stage_profile.get('tone', 'helpful and direct')}
{startup_section}
{meetings_section}{notes_section}{milestones_section}{schedule_section}{blockers_section}
GUIDELINES:
1. Always consider the founder's current stage when giving advice
2. Reference specific frameworks appropriate to their stage
3. Flag red flags if you notice anti-patterns: {', '.join(stage_profile.get('red_flags', []))}
4. Be concise but thorough — founders are time-poor
5. When suggesting tasks, tie them to specific milestones
6. Cross-reference information from meetings, notes, and contacts when relevant
7. Proactively suggest follow-ups and next steps
8. Challenge assumptions constructively — be a thinking partner, not a yes-machine
"""
