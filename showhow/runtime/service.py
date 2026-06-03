from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

from .process_manager import RecorderProcessManager
from .recorder_client import RecorderHTTPClient


class ShowHowService:
    """Parent-side orchestration for recorder and tutorial generation."""

    def __init__(
        self,
        recorder_client: Optional[RecorderHTTPClient] = None,
        process_manager: Optional[RecorderProcessManager] = None,
        tutorial_generator: Optional[Any] = None,
    ) -> None:
        self.recorder_client = recorder_client or RecorderHTTPClient()
        self.process_manager = process_manager or RecorderProcessManager(
            client=self.recorder_client
        )
        self.tutorial_generator = tutorial_generator
        self._last_session_id: Optional[str] = None
        self._last_data_path: Optional[str] = None
        self._session_data_paths: dict[str, str] = {}
        self._watchdog_interval = float(
            os.getenv("SHOWHOW_WATCHDOG_INTERVAL_SEC", "3.0")
        )
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

    def startup(self) -> dict[str, Any]:
        self.process_manager.start()
        self._start_watchdog()
        return {
            "status": "ok",
            "recorder_pid": self.process_manager.pid,
        }

    def shutdown(self) -> dict[str, Any]:
        self._stop_watchdog()
        self.process_manager.stop(force=True)
        return {"status": "stopped"}

    def ensure_recorder_healthy(self) -> None:
        if not self.process_manager.is_running():
            self.process_manager.start()
            return
        try:
            health = self.recorder_client.health()
            if health.get("status") != "ok":
                self.process_manager.restart()
        except Exception:
            self.process_manager.restart()

    def start_recording(
        self,
        topic: Optional[str] = None,
        folder: Optional[str] = None,
        base_name: Optional[str] = None,
    ) -> dict[str, Any]:
        self.ensure_recorder_healthy()
        result = self.recorder_client.start_recording(
            topic=topic, folder=folder, base_name=base_name
        )
        self._last_session_id = result.get("session_id")
        self._last_data_path = result.get("data_path")
        if self._last_session_id and self._last_data_path:
            self._session_data_paths[self._last_session_id] = self._last_data_path
        return result

    def stop_recording(self) -> dict[str, Any]:
        self.ensure_recorder_healthy()
        result = self.recorder_client.stop_recording()
        self._last_session_id = result.get("session_id")
        self._last_data_path = result.get("data_path")
        if self._last_session_id and self._last_data_path:
            self._session_data_paths[self._last_session_id] = self._last_data_path
        return result

    def generate_tutorial(
        self, session_id: str, data_path: Optional[str] = None
    ) -> dict[str, Any]:
        return self.generate_tutorial_with_options(
            session_id=session_id, data_path=data_path
        )

    def generate_tutorial_with_options(
        self,
        session_id: str,
        data_path: Optional[str] = None,
        caption_model: Optional[str] = None,
        planner_model: Optional[str] = None,
        critic_model: Optional[str] = None,
        on_stage: Any = None,
    ) -> dict[str, Any]:
        if not session_id:
            raise ValueError("session_id is required")
        use_custom_modes = any(
            v is not None for v in (caption_model, planner_model, critic_model)
        )
        if self.tutorial_generator is None or use_custom_modes:
            try:
                from showhow.tutorial_generator import (
                    TutorialGenerator as _TutorialGenerator,
                )
                from showhow.tutorial_generator.config import (
                    CriticConfig,
                    GeneratorConfig,
                    PlannerConfig,
                )

                selected_caption_model = caption_model or "gpt-4o"
                selected_planner_model = planner_model or selected_caption_model
                selected_critic_model = critic_model or selected_caption_model

                cfg = GeneratorConfig(
                    caption_model=selected_caption_model,
                    planner=PlannerConfig(model=selected_planner_model),
                    critic=CriticConfig(model=selected_critic_model),
                )
                gen = _TutorialGenerator(config=cfg)
                if use_custom_modes or self.tutorial_generator is None:
                    self.tutorial_generator = gen
            except Exception:
                self.tutorial_generator = None

        if self.tutorial_generator is None:
            return {
                "status": "not_implemented",
                "session_id": session_id,
                "message": "Tutorial generation is unavailable in the current runtime.",
                "data_path": self._last_data_path,
            }
        if data_path:
            resolved_override = str(Path(data_path).expanduser().resolve())
            self._session_data_paths[session_id] = resolved_override
            self._last_session_id = session_id
            self._last_data_path = resolved_override
        data_path = self._resolve_data_path(session_id=session_id)
        if not data_path:
            raise RuntimeError(
                f"Unable to resolve data path for session_id={session_id}"
            )

        video_path = self._find_video_path(Path(data_path))
        if not video_path:
            raise RuntimeError(
                f"No video file found under session data path: {data_path}. "
                "Recorder-service video capture must be integrated before tutorial generation."
            )

        result = self.tutorial_generator.generate(  # type: ignore[attr-defined]
            session_id=session_id,
            video_path=str(video_path),
            data_path=str(data_path),
            on_stage=on_stage,
        )
        report = getattr(result, "generation_report", {}) or {}
        return {
            "status": "success",
            "session_id": session_id,
            "video_path": str(result.video_path),
            "data_path": str(result.data_path),
            "trace_json_path": str(result.parse.trace_json_path),
            "draft_json_path": str(result.plan.draft_json_path),
            "tutorial_output_dir": str(result.compose.output_dir),
            "approved": result.plan.approved,
            "iterations": result.plan.iterations,
            "generation_report": report,
        }

    def stop_and_generate(self) -> dict[str, Any]:
        stop_result = self.stop_recording()
        session_id = stop_result.get("session_id")
        if not session_id:
            raise RuntimeError("Recorder stop result did not include session_id")
        generate_result = self.generate_tutorial_with_options(session_id=session_id)
        return {
            "stop": stop_result,
            "tutorial": generate_result,
        }

    def status(self) -> dict[str, Any]:
        self.ensure_recorder_healthy()
        status = self.recorder_client.status()
        status["recorder_pid"] = self.process_manager.pid
        return status

    def last_session_path(self) -> Optional[Path]:
        if not self._last_data_path:
            return None
        return Path(self._last_data_path)

    def _start_watchdog(self) -> None:
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, name="showhow-watchdog", daemon=True
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=1.5)
        self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        while not self._watchdog_stop.wait(self._watchdog_interval):
            try:
                self.ensure_recorder_healthy()
            except Exception:
                # Keep watchdog resilient; tool calls surface user-facing errors.
                pass

    def _find_video_path(self, data_path: Path) -> Optional[Path]:
        if not data_path.exists():
            return None
        candidates = sorted(data_path.glob("*.mkv"))
        if candidates:
            return candidates[0]
        recursive = sorted(data_path.glob("**/*.mkv"))
        if recursive:
            return recursive[0]
        return None

    def _resolve_data_path(self, session_id: str) -> Optional[str]:
        known = self._session_data_paths.get(session_id)
        if known:
            return known
        if self._last_session_id == session_id and self._last_data_path:
            return self._last_data_path

        candidate = self._default_record_root() / session_id
        if candidate.exists():
            resolved = str(candidate.resolve())
            self._session_data_paths[session_id] = resolved
            return resolved
        return None

    def _default_record_root(self) -> Path:
        configured = os.getenv("SHOWHOW_RECORD_ROOT", "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return (Path.home() / "Downloads" / "record_save").resolve()
