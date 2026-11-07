"""ArSL Studio — FastAPI Backend Application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.dataset import router as dataset_router
from backend.api.gestures import router as gestures_router
from backend.api.models import router as models_router
from backend.api.stream import router as stream_router
from backend.api.system import router as system_router
from backend.api.training import router as training_router
from backend.services.dataset_service import DatasetService
from backend.services.recognition_service import RecognitionService
from backend.services.training_service import TrainingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("arsl.backend")

# Initialize shared singleton services
dataset_service = DatasetService()
recognition_service = RecognitionService()
training_service = TrainingService()

app = FastAPI(
    title="ArSL Studio API",
    description="Local web backend for Arabic Sign Language Recognition",
    version="2.0.0",
)

# CORS configuration for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(system_router)
app.include_router(gestures_router)
app.include_router(dataset_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(stream_router)

# Mount React static frontend if built
WEB_DIST_DIR = PROJECT_ROOT / "web" / "dist"
if WEB_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        # Don't intercept API or WebSocket routes
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            return None
        file_path = WEB_DIST_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        index_file = WEB_DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"error": "Frontend not found"}
else:
    logger.info("web/dist not found. Run 'npm run build' inside web/ to enable single-server production mode.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

