"""Gestures CRUD endpoints."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.schemas.models import GestureCreate, GestureResponse, GestureUpdate
from backend.services.dataset_service import DatasetService

router = APIRouter(prefix="/api/gestures", tags=["gestures"])


def get_dataset_service() -> DatasetService:
    from backend.main import dataset_service
    return dataset_service


@router.get("", response_model=list[GestureResponse])
def list_gestures(
    include_archived: bool = Query(False),
    service: DatasetService = Depends(get_dataset_service),
) -> Any:
    return service.list_gestures(include_archived=include_archived)


@router.post("", response_model=GestureResponse, status_code=201)
def create_gesture(
    payload: GestureCreate,
    service: DatasetService = Depends(get_dataset_service),
) -> Any:
    try:
        return service.add_gesture(
            name=payload.name,
            category=payload.category,
            description=payload.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create gesture: {e}")


@router.get("/{gesture_id}", response_model=GestureResponse)
def get_gesture(
    gesture_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> Any:
    g = service.get_gesture(gesture_id)
    if not g:
        raise HTTPException(status_code=404, detail="Gesture not found")
    return g


@router.patch("/{gesture_id}", response_model=GestureResponse)
def update_gesture(
    gesture_id: str,
    payload: GestureUpdate,
    service: DatasetService = Depends(get_dataset_service),
) -> Any:
    try:
        if payload.name:
            res = service.rename_gesture(gesture_id, payload.name)
        else:
            res = service.get_gesture(gesture_id)
        if not res:
            raise HTTPException(status_code=404, detail="Gesture not found")
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{gesture_id}/archive")
def archive_gesture(
    gesture_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> dict[str, bool]:
    ok = service.archive_gesture(gesture_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Gesture not found or could not be archived")
    return {"success": True}


@router.post("/{gesture_id}/restore")
def restore_gesture(
    gesture_id: str,
    service: DatasetService = Depends(get_dataset_service),
) -> dict[str, bool]:
    ok = service.restore_gesture(gesture_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Gesture not found in archive")
    return {"success": True}


@router.delete("/{gesture_id}")
def delete_gesture(
    gesture_id: str,
    purge_samples: bool = Query(False),
    service: DatasetService = Depends(get_dataset_service),
) -> dict[str, bool]:
    ok = service.delete_gesture(gesture_id, purge_samples=purge_samples)
    if not ok:
        raise HTTPException(status_code=404, detail="Gesture not found")
    return {"success": True}

