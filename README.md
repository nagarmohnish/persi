# Persi — AI Founder Assistant

Your AI-powered startup co-pilot. From idea to launch.

## Architecture

- **Frontend**: Next.js 14 + Tailwind CSS + shadcn/ui
- **Backend**: FastAPI + SQLAlchemy (async)
- **Database**: PostgreSQL
- **AI**: Claude API (Phase 2)

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 20+
- Python 3.11+

### Local Development

```bash
# 1. Start PostgreSQL
docker compose up db -d

# 2. Backend
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head           # Run migrations
uvicorn app.main:app --reload  # Start on :8000

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                    # Start on :3000
```

Open http://localhost:3000 to use Persi.

## Project Structure

```
persi/
├── backend/          # FastAPI API
│   ├── app/          # Application code
│   │   ├── models/   # SQLAlchemy ORM models
│   │   ├── schemas/  # Pydantic request/response models
│   │   └── routers/  # API endpoint routers
│   └── alembic/      # Database migrations
├── frontend/         # Next.js 14 app
│   ├── app/          # Pages (App Router)
│   ├── components/   # React components
│   └── lib/          # Utilities
├── data/             # PG essays + knowledge base
└── scripts/          # Data processing scripts
```

## API Docs

With the backend running, visit http://localhost:8000/docs for interactive Swagger docs.
