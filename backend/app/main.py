import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import engine
from app.models import Base
from app.routers import health, conversations, profile, legacy, meetings, notes, calendar_routes, travel, journey, contacts, ai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Persi API starting up")
    yield
    await engine.dispose()
    logger.info("Persi API shut down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc" if settings.debug else None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_url, "http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router, tags=["health"])
    app.include_router(
        conversations.router, prefix="/api/v1/conversations", tags=["conversations"]
    )
    app.include_router(profile.router, prefix="/api/v1/profile", tags=["profile"])
    app.include_router(legacy.router, tags=["legacy"])
    app.include_router(meetings.router, prefix="/api/v1/meetings", tags=["meetings"])
    app.include_router(notes.router, prefix="/api/v1/notes", tags=["notes"])
    app.include_router(calendar_routes.router, prefix="/api/v1/calendar", tags=["calendar"])
    app.include_router(travel.router, prefix="/api/v1/travel", tags=["travel"])
    app.include_router(journey.router, prefix="/api/v1/journey", tags=["journey"])
    app.include_router(contacts.router, prefix="/api/v1/contacts", tags=["contacts"])
    app.include_router(ai.router, prefix="/api/v1/ai", tags=["ai"])

    return app


app = create_app()
