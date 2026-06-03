from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StartRequest(BaseModel):
    topic: Optional[str] = Field(default=None, description="Optional recording topic")
    folder: Optional[str] = Field(
        default=None, description="Optional output root folder"
    )
    base_name: Optional[str] = Field(
        default=None, description="Optional session name prefix"
    )


class StartResponse(BaseModel):
    status: str = "success"
    session_id: str
    data_path: str
    video_path: Optional[str] = None
    input_capture_enabled: bool = True


class StopResponse(BaseModel):
    status: str = "success"
    session_id: str
    data_path: str
    event_log_path: str
    metadata_path: str
    event_count: int
    video_path: Optional[str] = None
    input_capture_enabled: bool = True


class StatusResponse(BaseModel):
    recording: bool
    session_id: Optional[str] = None
    topic: Optional[str] = None
    data_path: Optional[str] = None
    video_path: Optional[str] = None
    input_capture_enabled: bool = True
    event_count: int = 0
    duration: float = 0.0


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str


class SessionPaths(BaseModel):
    root: Path
    events_path: Path
    metadata_path: Path
    video_path: Path
