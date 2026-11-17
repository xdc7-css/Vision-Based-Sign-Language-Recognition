"""FastAPI Endpoints for KArSL Scientific Research (SVM vs CNN)."""

from __future__ import annotations

from typing import Any
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
    name: str
    notes: str = ""


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


