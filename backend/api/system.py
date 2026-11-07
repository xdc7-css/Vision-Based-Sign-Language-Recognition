"""System telemetry, health, and sentence builder endpoints."""

from __future__ import annotations

import psutil
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.schemas.models import SystemStatusResponse
from backend.services.dataset_service import DatasetService
from backend.services.recognition_service import RecognitionService

router = APIRouter(prefix="/api", tags=["system"])


class SentenceActionRequest(BaseModel):
    word: str = ""


def get_dataset_service() -> DatasetService:
    from backend.main import dataset_service
    return dataset_service


def get_recognition_service() -> RecognitionService:
    from backend.main import recognition_service
    return recognition_service


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/system/status", response_model=SystemStatusResponse)
def get_system_status(
    rec_svc: RecognitionService = Depends(get_recognition_service),
    ds_svc: DatasetService = Depends(get_dataset_service),
) -> SystemStatusResponse:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    active_m = rec_svc.get_active_model_info()
    gestures = ds_svc.list_gestures()

    return SystemStatusResponse(
        status="ok",
        version="2.0.0",
        cpu_percent=round(cpu, 1),
        memory_percent=round(mem, 1),
        active_model=f"{active_m['name']} ({active_m['num_classes']} signs)",
        total_gestures=len(gestures),
    )


# ── Sentence Builder Endpoints ───────────────────────────────────────────

@router.post("/sentence/word")
def add_word(
    payload: SentenceActionRequest,
    service: RecognitionService = Depends(get_recognition_service),
) -> dict[str, str]:
    full_sentence = service.add_sentence_word(payload.word)
    return {"sentence": full_sentence}


@router.post("/sentence/delete")
def delete_last_word(service: RecognitionService = Depends(get_recognition_service)) -> dict[str, str]:
    full_sentence = service.delete_last_word()
    return {"sentence": full_sentence}


@router.post("/sentence/clear")
def clear_sentence(service: RecognitionService = Depends(get_recognition_service)) -> dict[str, str]:
    full_sentence = service.clear_sentence()
    return {"sentence": full_sentence}


@router.get("/sentence")
def get_sentence(service: RecognitionService = Depends(get_recognition_service)) -> dict[str, str]:
    return {"sentence": service.get_sentence()}

