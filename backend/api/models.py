"""Model management endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends

from backend.schemas.models import ActiveModelResponse
from backend.services.recognition_service import RecognitionService

router = APIRouter(prefix="/api/models", tags=["models"])


def get_recognition_service() -> RecognitionService:
    from backend.main import recognition_service
    return recognition_service


@router.get("/active", response_model=ActiveModelResponse)
def get_active_model(service: RecognitionService = Depends(get_recognition_service)) -> Any:
    return service.get_active_model_info()


@router.post("/reload")
def reload_active_model(service: RecognitionService = Depends(get_recognition_service)) -> dict[str, bool]:
    ok = service.reload_model()
    return {"success": ok}

