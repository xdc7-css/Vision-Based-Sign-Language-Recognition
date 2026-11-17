"""FastAPI Endpoints for KArSL Scientific Research (SVM vs CNN)."""

from __future__ import annotations

import base64
from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.research_service import ResearchService

router = APIRouter(prefix="/api/research", tags=["research"])
research_service = ResearchService()


class PrepareDatasetRequest(BaseModel):
    samples_per_sign_per_signer: int = 15
    force: bool = False


class TrainSVMRequest(BaseModel):
    kernel: str = "rbf"
    C: float = 2.0


class TrainCNNRequest(BaseModel):
    epochs: int = 35
    batch_size: int = 16


class ComparisonRequest(BaseModel):
    fold: str = "test_Signer03"


class RegisterVolunteerRequest(BaseModel):
    name: str = ""
    notes: str = ""


class UploadRecordingRequest(BaseModel):
    volunteer_id: str
    sign_id: str
    rep_idx: int
    is_idle: bool = False
    video_base64: str
    filename: Optional[str] = "recording.webm"
    fps: float = 30.0
    camera_device: Optional[str] = None
    notes: Optional[str] = None


@router.get("/status")
def get_research_status() -> dict[str, Any]:
    """Get research dataset, model checkpoints, and comparison report status."""
    return research_service.get_status()


@router.post("/prepare-dataset")
def prepare_research_dataset(req: PrepareDatasetRequest) -> dict[str, Any]:
    """Prepare research dataset from KArSL data."""
    try:
        return research_service.prepare_dataset(
            samples_per_sign_per_signer=req.samples_per_sign_per_signer,
            force=req.force,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-svm")
def train_research_svm(req: TrainSVMRequest) -> dict[str, Any]:
    """Train Model A (SVM) on KArSL dataset."""
    try:
        return research_service.train_svm(kernel=req.kernel, C=req.C)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-cnn")
def train_research_cnn(req: TrainCNNRequest) -> dict[str, Any]:
    """Train Model B (1D Temporal CNN) on KArSL dataset."""
    try:
        return research_service.train_cnn(epochs=req.epochs, batch_size=req.batch_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
def run_scientific_comparison(req: ComparisonRequest) -> dict[str, Any]:
    """Execute scientific evaluation comparing SVM vs CNN on held-out test signer."""
    try:
        return research_service.run_comparison(fold=req.fold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison")
def get_latest_comparison() -> dict[str, Any]:
    """Retrieve existing comparison report."""
    comp = research_service.evaluator.get_latest_comparison()
    if not comp:
        raise HTTPException(status_code=404, detail="No scientific comparison has been executed yet.")
    return comp


# ─── Independent Human Evaluation Endpoints ─────────────────────────────────

@router.get("/independent-eval/status")
def get_independent_eval_status() -> dict[str, Any]:
    """Get current status of independent human evaluation."""
    return research_service.get_independent_eval_status()


@router.get("/independent-eval/volunteers")
def list_independent_volunteers() -> list[dict[str, Any]]:
    """List registered volunteers for independent evaluation."""
    return research_service.list_independent_volunteers()


@router.post("/independent-eval/volunteers")
def register_independent_volunteer(req: RegisterVolunteerRequest) -> dict[str, Any]:
    """Register a new human volunteer for independent testing."""
    try:
        return research_service.register_independent_volunteer(name=req.name, notes=req.notes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/independent-eval/volunteers/{volunteer_id}/progress")
def get_volunteer_progress(volunteer_id: str) -> dict[str, Any]:
    """Get recording progress for a given volunteer."""
    try:
        return research_service.get_volunteer_progress(volunteer_id=volunteer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/independent-eval/evaluate")
def run_independent_evaluation() -> dict[str, Any]:
    """Execute frozen model evaluation on independent volunteer recordings."""
    try:
        return research_service.run_independent_evaluation()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/independent-eval/consent")
def get_independent_eval_consent() -> dict[str, Any]:
    """Get research informed consent notice for volunteer recording."""
    return research_service.get_consent_notice()


@router.get("/independent-eval/signs")
def list_independent_eval_signs() -> list[dict[str, Any]]:
    """List 10 predefined Arabic digit signs with instructions."""
    return research_service.list_sign_references()


@router.get("/independent-eval/signs/{sign_id}")
def get_independent_eval_sign_info(sign_id: str) -> dict[str, Any]:
    """Get instructions for a specific sign ID or 'idle'."""
    info = research_service.get_sign_reference_info(sign_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Sign ID '{sign_id}' not found.")
    return info


@router.get("/independent-eval/volunteers/{volunteer_id}/next-rep/{sign_id}")
def get_next_repetition_index(volunteer_id: str, sign_id: str) -> dict[str, Any]:
    """Get next non-duplicate repetition index for a volunteer and sign."""
    next_idx = research_service.get_next_repetition_index(volunteer_id, sign_id)
    return {
        "volunteer_id": volunteer_id,
        "sign_id": sign_id,
        "next_rep_idx": next_idx,
    }


@router.delete("/independent-eval/volunteers/{volunteer_id}")
def delete_independent_volunteer(volunteer_id: str) -> dict[str, Any]:
    """Permanently delete volunteer records and video files for privacy."""
    try:
        return research_service.delete_volunteer(volunteer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/independent-eval/recordings/{record_id}")
def delete_independent_recording(record_id: str) -> dict[str, Any]:
    """Delete an individual recording from manifest and disk."""
    try:
        res = research_service.delete_recording(record_id)
        if not res.get("success"):
            raise HTTPException(status_code=404, detail=res.get("error", "Not found"))
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/independent-eval/recordings/upload")
def upload_independent_recording(req: UploadRecordingRequest) -> dict[str, Any]:
    """Upload a recorded video (base64 encoded) from the web camera recording interface."""
    try:
        b64_str = req.video_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        content = base64.b64decode(b64_str)
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

        entry = research_service.save_recording_from_bytes(
            volunteer_id=req.volunteer_id,
            sign_id=req.sign_id,
            rep_idx=req.rep_idx,
            video_bytes=content,
            filename=req.filename or "recording.webm",
            is_idle=req.is_idle,
        )
        return {"success": True, "recording": entry, "record": entry}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



