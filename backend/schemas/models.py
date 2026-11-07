"""Pydantic schemas for the ArSL Studio FastAPI backend."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Gestures ─────────────────────────────────────────────────────────────

class GestureCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Arabic gesture name")
    category: str = Field("static", pattern="^(static|dynamic|both|alphabet)$")
    description: str = Field("", max_length=500)


class GestureUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    category: Optional[str] = Field(None, pattern="^(static|dynamic|both|alphabet)$")
    description: Optional[str] = Field(None, max_length=500)


class GestureResponse(BaseModel):
    id: str
    name: str
    category: str
    description: str = ""
    is_active: bool = True
    is_default: bool = False
    sample_count: int = 0
    is_ready: bool = False
    created_at: float = 0.0
    updated_at: float = 0.0


# ── Dataset ──────────────────────────────────────────────────────────────

class SampleInfo(BaseModel):
    sample_id: str
    timestamp: float
    schema_version: str = "v2_multimodal"
    num_frames: int = 1
    has_face: bool = False
    has_left: bool = False
    has_right: bool = False
    handedness: str = ""


class DatasetSummaryResponse(BaseModel):
    total_classes: int
    total_samples: int
    trainable_classes: int
    multimodal_samples: int
    legacy_samples: int
    classes: list[dict[str, Any]]


# ── Training ─────────────────────────────────────────────────────────────

class TrainingReadinessResponse(BaseModel):
    is_trainable: bool
    trainable_classes_count: int
    total_multimodal_samples: int
    accepted_classes: dict[str, int]
    insufficient_classes: dict[str, int]
    model_status: str  # "Ready", "Outdated", "Untrained"
    reason: Optional[str] = None
    last_trained_timestamp: Optional[float] = None


class TrainingStartRequest(BaseModel):
    model_type: str = Field("mlp", pattern="^(mlp|svm|rf)$")
    epochs: int = Field(60, ge=5, le=500)
    batch_size: int = Field(16, ge=4, le=128)
    learning_rate: float = Field(0.001, gt=0, le=0.1)
    val_split: float = Field(0.2, ge=0.1, le=0.5)
    allowed_classes: Optional[list[str]] = None


class TrainingJobStatus(BaseModel):
    job_id: str
    status: str  # "idle", "running", "completed", "failed", "cancelled"
    progress: float = 0.0
    current_epoch: int = 0
    max_epochs: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    metrics: Optional[dict[str, Any]] = None
    error: Optional[str] = None


# ── Models ───────────────────────────────────────────────────────────────

class ActiveModelResponse(BaseModel):
    name: str
    model_type: str
    classes: list[str]
    num_classes: int
    accuracy: Optional[float] = None
    is_trained: bool = False
    updated_at: Optional[float] = None


# ── System ───────────────────────────────────────────────────────────────

class SystemStatusResponse(BaseModel):
    status: str = "ok"
    version: str = "2.0.0"
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    active_model: str = ""
    total_gestures: int = 0

