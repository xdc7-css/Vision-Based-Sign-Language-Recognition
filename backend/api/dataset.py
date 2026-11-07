"""Dataset inspection and sample management endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from backend.schemas.models import DatasetSummaryResponse, SampleInfo
from backend.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def get_dataset_service() -> DatasetService:
    from backend.main import dataset_service
    return dataset_service


@router.get("/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary(service: DatasetService = Depends(get_dataset_service)) -> Any:
    return service.get_summary()


@router.get("/gestures/{gesture_id}/samples", response_model=list[SampleInfo])
def list_samples_for_gesture(
    gesture_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> Any:
    return service.list_samples_for_gesture(gesture_id)


@router.delete("/gestures/{gesture_id}/samples/{sample_id}")
def delete_sample(
    gesture_id: str,
    sample_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> dict[str, bool]:
    ok = service.delete_sample(gesture_id, sample_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Sample not found or could not be deleted")
    return {"success": True}

