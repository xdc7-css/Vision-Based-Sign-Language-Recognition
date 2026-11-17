"""Model management endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends

from backend.schemas.models import ActiveModelResponse, ModelSelectRequest
from backend.services.recognition_service import RecognitionService

router = APIRouter(prefix="/api/models", tags=["models"])


def get_recognition_service() -> RecognitionService:
    from backend.main import recognition_service
    return recognition_service


@router.get("/active", response_model=ActiveModelResponse)
def get_active_model(service: RecognitionService = Depends(get_recognition_service)) -> Any:
    return service.get_active_model_info()


@router.get("/available")
def get_available_models(service: RecognitionService = Depends(get_recognition_service)) -> list[dict[str, Any]]:
    return service.get_available_models()


@router.post("/select")
def select_active_model(req: ModelSelectRequest, service: RecognitionService = Depends(get_recognition_service)) -> dict[str, Any]:
    ok = service.set_active_model(req.model_id)
    return {"success": ok, "active_model": service.get_active_model_info()}


@router.post("/reload")
def reload_active_model(service: RecognitionService = Depends(get_recognition_service)) -> dict[str, bool]:
    ok = service.reload_model()
    return {"success": ok}

