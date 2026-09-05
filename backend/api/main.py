"""
BirdFrame FastAPI application.

Start the server with:
    uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000

Or via the helper script:
    python -m backend.api.main
"""

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from backend.database.engine import init_db
from backend.api.routes import health, detections, species, stats, collage

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BirdFrame API starting up…")
    init_db()
    logger.info("Database initialised.")
    yield
    logger.info("BirdFrame API shutting down.")


app = FastAPI(
    title="BirdFrame API",
    description=(
        "REST API for BirdFrame — a system that listens for bird sounds "
        "and builds a visual record of detected species."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(detections.router, prefix="/api", tags=["Detections"])
app.include_router(species.router, prefix="/api", tags=["Species"])
app.include_router(stats.router, prefix="/api", tags=["Stats"])
app.include_router(collage.router, prefix="/api", tags=["Collage"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
    )