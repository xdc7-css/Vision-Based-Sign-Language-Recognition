"""Model training endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from backend.schemas.models import TrainingJobStatus, TrainingReadinessResponse, TrainingStartRequest
from backend.services.recognition_service import RecognitionService
from backend.services.training_service import TrainingService

router = APIRouter(prefix="/api/training", tags=["training"])


def get_training_service() -> TrainingService:
    from backend.main import training_service
    return training_service


def get_recognition_service() -> RecognitionService:
    from backend.main import recognition_service
    return recognition_service


@router.get("/readiness", response_model=TrainingReadinessResponse)
def get_training_readiness(service: TrainingService = Depends(get_training_service)) -> Any:
    return service.get_readiness()


@router.post("/start", response_model=TrainingJobStatus)
def start_training_job(
    payload: TrainingStartRequest,
    train_svc: TrainingService = Depends(get_training_service),
    rec_svc: RecognitionService = Depends(get_recognition_service),
) -> Any:
    try:
        job = train_svc.start_training(
            model_type=payload.model_type,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
            val_split=payload.val_split,
            allowed_classes=payload.allowed_classes,
            on_complete_callback=lambda: rec_svc.reload_model(),
        )
        return job.to_dict()
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {e}")


@router.get("/jobs/{job_id}", response_model=TrainingJobStatus)
def get_job_status(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
) -> Any:
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job.to_dict()


@router.post("/jobs/{job_id}/cancel")
def cancel_training_job(
    job_id: str,
    service: TrainingService = Depends(get_training_service),
) -> dict[str, bool]:
    ok = service.cancel_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    return {"success": True}

