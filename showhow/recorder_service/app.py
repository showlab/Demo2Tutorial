from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException

from .engine import RecorderEngine
from .models import (
    ErrorResponse,
    StartRequest,
    StartResponse,
    StatusResponse,
    StopResponse,
)


def create_app(engine: RecorderEngine | None = None) -> FastAPI:
    recorder = engine or RecorderEngine()
    app = FastAPI(title="ShowHow Recorder Service", version="0.1.0")

    @app.on_event("shutdown")
    def _shutdown_cleanup() -> None:
        if recorder.is_recording:
            try:
                recorder.stop()
            except Exception:
                pass

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        return StatusResponse(**recorder.status())

    @app.post(
        "/start",
        response_model=StartResponse,
        responses={409: {"model": ErrorResponse}},
    )
    def start(req: Optional[StartRequest] = None) -> StartResponse:
        payload = req or StartRequest()
        try:
            result = recorder.start(
                topic=payload.topic, folder=payload.folder, base_name=payload.base_name
            )
            return StartResponse(**result)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise HTTPException(
                status_code=500, detail=f"Failed to start recording: {exc}"
            ) from exc

    @app.post(
        "/stop", response_model=StopResponse, responses={409: {"model": ErrorResponse}}
    )
    def stop() -> StopResponse:
        try:
            result = recorder.stop()
            return StopResponse(**result)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise HTTPException(
                status_code=500, detail=f"Failed to stop recording: {exc}"
            ) from exc

    # Legacy aliases to ease migration from Recorder2 callers.
    app.add_api_route(
        "/startRecording", start, methods=["POST"], response_model=StartResponse
    )
    app.add_api_route(
        "/stopRecording", stop, methods=["POST"], response_model=StopResponse
    )

    return app
