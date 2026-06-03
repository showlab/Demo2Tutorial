from __future__ import annotations

import json
import os
import platform
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from pynput import keyboard, mouse

from .models import SessionPaths
from .platform_context import create_context_provider
from .video_capture import FFmpegVideoRecorder


@dataclass
class SessionState:
    session_id: str
    topic: Optional[str]
    started_at_monotonic: float
    started_at_utc: str
    paths: SessionPaths
    event_count: int = 0


class RecorderEngine:
    """Records keyboard/mouse input events and stores them into a session folder."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._keyboard_listener: Optional[keyboard.Listener] = None
        self._mouse_listener: Optional[mouse.Listener] = None
        self._event_file = None
        self._session: Optional[SessionState] = None
        self._video_recorder: Optional[FFmpegVideoRecorder] = None
        self._video_capture_enabled = os.getenv(
            "SHOWHOW_ENABLE_VIDEO", "1"
        ).strip() not in {"0", "false", "False"}
        self._input_capture_enabled = True
        self._context_provider = create_context_provider()

        self._modifier_keys = {
            getattr(keyboard.Key, name)
            for name in (
                "ctrl",
                "ctrl_l",
                "ctrl_r",
                "cmd",
                "cmd_l",
                "cmd_r",
                "alt",
                "alt_l",
                "alt_r",
                "shift",
                "shift_l",
                "shift_r",
            )
            if hasattr(keyboard.Key, name)
        }
        self._pressed_modifiers: set[Any] = set()

        self._drag_start: Optional[tuple[int, int]] = None
        self._is_dragging = False
        self._context_cache_t: float = 0.0
        self._context_cache: dict[str, Any] = {}
        self._context_cache_interval_sec = 0.3
        self._screen_metadata: dict[str, Any] = {}

    @property
    def is_recording(self) -> bool:
        return self._session is not None

    def status(self) -> dict[str, Any]:
        with self._lock:
            if not self._session:
                return {
                    "recording": False,
                    "session_id": None,
                    "topic": None,
                    "data_path": None,
                    "video_path": None,
                    "input_capture_enabled": self._input_capture_enabled,
                    "event_count": 0,
                    "duration": 0.0,
                }
            duration = max(0.0, time.monotonic() - self._session.started_at_monotonic)
            return {
                "recording": True,
                "session_id": self._session.session_id,
                "topic": self._session.topic,
                "data_path": str(self._session.paths.root),
                "video_path": str(self._session.paths.video_path),
                "input_capture_enabled": self._input_capture_enabled,
                "event_count": self._session.event_count,
                "duration": round(duration, 3),
            }

    def start(
        self,
        topic: Optional[str] = None,
        folder: Optional[str] = None,
        base_name: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._session:
                raise RuntimeError("Recording already in progress")

            session_id = self._build_session_id(topic=topic, base_name=base_name)
            paths = self._create_session_paths(session_id=session_id, folder=folder)
            started_at_utc = datetime.now(timezone.utc).isoformat()
            self._screen_metadata = self._capture_screen_metadata()

            self._event_file = paths.events_path.open("a", encoding="utf-8")
            self._session = SessionState(
                session_id=session_id,
                topic=topic,
                started_at_monotonic=time.monotonic(),
                started_at_utc=started_at_utc,
                paths=paths,
            )

            self._write_metadata(
                {
                    "session_id": session_id,
                    "topic": topic,
                    "started_at": started_at_utc,
                    "platform": platform.platform(),
                    "status": "recording",
                    "data_path": str(paths.root),
                    "event_log_path": str(paths.events_path),
                    "video_path": str(paths.video_path),
                    "input_capture_enabled": self._input_capture_enabled,
                    "screen": self._screen_metadata,
                }
            )

            try:
                self._start_video_capture(paths.video_path)
                self._start_listeners()
            except Exception:
                # Ensure partially-created session state is fully rolled back.
                if self._event_file:
                    self._event_file.close()
                    self._event_file = None
                self._stop_video_capture()
                self._session = None
                try:
                    shutil.rmtree(paths.root, ignore_errors=True)
                except Exception:
                    pass
                raise

            return {
                "session_id": session_id,
                "data_path": str(paths.root),
                "video_path": str(paths.video_path),
                "input_capture_enabled": self._input_capture_enabled,
            }

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._session:
                raise RuntimeError("No active recording")

            session = self._session
            duration = max(0.0, time.monotonic() - session.started_at_monotonic)

            self._stop_listeners()

            if self._event_file:
                self._event_file.flush()
                self._event_file.close()
                self._event_file = None

            metadata = {
                "session_id": session.session_id,
                "topic": session.topic,
                "started_at": session.started_at_utc,
                "stopped_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(duration, 3),
                "platform": platform.platform(),
                "status": "stopped",
                "data_path": str(session.paths.root),
                "event_log_path": str(session.paths.events_path),
                "video_path": str(session.paths.video_path),
                "input_capture_enabled": self._input_capture_enabled,
                "event_count": session.event_count,
                "screen": self._screen_metadata,
            }
            self._write_metadata(metadata)

            result = {
                "session_id": session.session_id,
                "data_path": str(session.paths.root),
                "event_log_path": str(session.paths.events_path),
                "metadata_path": str(session.paths.metadata_path),
                "event_count": session.event_count,
                "video_path": str(session.paths.video_path),
                "input_capture_enabled": self._input_capture_enabled,
            }

            # Clear session state NOW so status() immediately returns recording=False,
            # even while we are still waiting for ffmpeg to finalize the file.
            self._session = None
            self._pressed_modifiers.clear()
            self._drag_start = None
            self._is_dragging = False
            self._screen_metadata = {}

            # Grab the video recorder reference before releasing the lock.
            video_recorder = self._video_recorder
            self._video_recorder = None

        # Stop video OUTSIDE the lock — this may take a few seconds (ffmpeg flush +
        # optional remux).  The lock is now free so concurrent status() calls return
        # immediately instead of blocking.
        if video_recorder:
            try:
                video_recorder.stop()
            except Exception:
                pass

        return result

    def _default_output_root(self) -> Path:
        configured_root = os.getenv("SHOWHOW_RECORD_ROOT", "").strip()
        if configured_root:
            return Path(configured_root).expanduser()
        return Path.home() / "Downloads" / "record_save"

    def _build_session_id(self, topic: Optional[str], base_name: Optional[str]) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = (base_name or topic or "session").strip().replace(" ", "_")
        safe_prefix = (
            "".join(ch for ch in prefix if ch.isalnum() or ch in {"_", "-"})
            or "session"
        )
        return f"{safe_prefix}_{stamp}_{uuid4().hex[:8]}"

    def _create_session_paths(
        self, session_id: str, folder: Optional[str]
    ) -> SessionPaths:
        root = (
            Path(folder).expanduser().resolve()
            if folder
            else self._default_output_root().resolve()
        )
        root.mkdir(parents=True, exist_ok=True)

        session_root = root / session_id
        session_root.mkdir(parents=True, exist_ok=False)

        return SessionPaths(
            root=session_root,
            events_path=session_root / "events.jsonl",
            metadata_path=session_root / "metadata.json",
            video_path=session_root / f"{session_id}.mkv",
        )

    def _write_metadata(self, payload: dict[str, Any]) -> None:
        if not self._session:
            return
        self._session.paths.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _capture_screen_metadata(self) -> dict[str, Any]:
        if platform.system() != "Darwin":
            return {}
        screen_number = int(os.getenv("SHOWHOW_SCREEN_NUMBER", "0"))

        try:
            from AppKit import NSScreen

            screens = list(NSScreen.screens() or [])
            if screens:
                idx = min(max(0, screen_number), len(screens) - 1)
                screen = screens[idx]
                frame = screen.frame()
                scale = float(screen.backingScaleFactor() or 1.0)
                logical_w = int(round(float(frame.size.width)))
                logical_h = int(round(float(frame.size.height)))
                return {
                    "platform": "darwin",
                    "screen_number": idx,
                    "scale_factor": scale,
                    "coordinate_space": {
                        "width": logical_w,
                        "height": logical_h,
                        "origin_x": int(round(float(frame.origin.x))),
                        "origin_y": int(round(float(frame.origin.y))),
                        "units": "points",
                    },
                    "capture_space": {
                        "width": int(round(logical_w * scale)),
                        "height": int(round(logical_h * scale)),
                        "units": "pixels",
                    },
                }
        except Exception:
            pass

        return {}

    def _start_listeners(self) -> None:
        self._input_capture_enabled = self._is_input_capture_trusted()
        if not self._input_capture_enabled:
            return
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_move=self._on_move,
            on_scroll=self._on_scroll,
        )
        self._keyboard_listener.start()
        self._mouse_listener.start()

    def _stop_listeners(self) -> None:
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None

    def _is_input_capture_trusted(self) -> bool:
        try:
            return bool(self._context_provider.is_input_capture_trusted())
        except Exception:
            return True

    def _start_video_capture(self, video_path: Path) -> None:
        if not self._video_capture_enabled:
            return
        recorder = FFmpegVideoRecorder(output_path=video_path)
        try:
            recorder.start()
            self._video_recorder = recorder
        except Exception as exc:
            # Video capture can fail due to permissions; event capture still runs.
            import logging

            logging.getLogger(__name__).warning("Video capture failed: %s", exc)
            self._video_recorder = None

    def _stop_video_capture(self) -> None:
        if self._video_recorder:
            try:
                self._video_recorder.stop()
            finally:
                self._video_recorder = None

    def _elapsed_ms(self) -> int:
        if not self._session:
            return 0
        return int((time.monotonic() - self._session.started_at_monotonic) * 1000)

    def _active_window(self) -> Optional[str]:
        ctx = self._active_context()
        app_name = ctx.get("app_name")
        if app_name:
            return str(app_name)
        return None

    def _active_context(self) -> dict[str, Any]:
        now = time.monotonic()
        if (
            self._context_cache
            and (now - self._context_cache_t) < self._context_cache_interval_sec
        ):
            return dict(self._context_cache)

        try:
            context = self._context_provider.capture_active_context(
                input_capture_enabled=self._input_capture_enabled
            )
        except Exception:
            context = {
                "app_name": None,
                "bundle_id": None,
                "pid": None,
                "window_title": None,
                "url": None,
                "ui": None,
            }
        self._context_cache = context
        self._context_cache_t = now
        return dict(context)

    def _write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        with self._lock:
            if not self._session or not self._event_file:
                return

            context = self._active_context()
            event = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": self._elapsed_ms(),
                "event_type": event_type,
                "window": context.get("app_name"),
                "window_title": context.get("window_title"),
                "bundle_id": context.get("bundle_id"),
                "url": context.get("url"),
                "pid": context.get("pid"),
                "ui": context.get("ui"),
                "context": context,
                "payload": payload,
            }
            self._event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._session.event_count += 1

    def _key_name(self, key: Any) -> str:
        if hasattr(key, "char") and key.char:
            return str(key.char)
        if hasattr(key, "name") and key.name:
            return str(key.name)
        return str(key)

    def _current_modifiers(self) -> list[str]:
        return sorted(self._key_name(k) for k in self._pressed_modifiers)

    def _on_key_press(self, key: Any) -> None:
        if key in self._modifier_keys:
            self._pressed_modifiers.add(key)
        self._write_event(
            "key_press",
            {
                "key": self._key_name(key),
                "modifiers": self._current_modifiers(),
            },
        )

    def _on_key_release(self, key: Any) -> None:
        self._write_event(
            "key_release",
            {
                "key": self._key_name(key),
                "modifiers": self._current_modifiers(),
            },
        )
        if key in self._modifier_keys:
            self._pressed_modifiers.discard(key)

    def _on_click(self, x: int, y: int, button: mouse.Button, pressed: bool) -> None:
        button_name = str(button).split(".")[-1]
        if pressed:
            if button == mouse.Button.left:
                self._drag_start = (int(x), int(y))
                self._is_dragging = False
            self._write_event(
                "mouse_down", {"x": int(x), "y": int(y), "button": button_name}
            )
            return

        if self._is_dragging and self._drag_start is not None:
            self._write_event(
                "mouse_drag_end",
                {
                    "x": int(x),
                    "y": int(y),
                    "button": button_name,
                    "drag_start": {"x": self._drag_start[0], "y": self._drag_start[1]},
                },
            )
        else:
            self._write_event(
                "mouse_up", {"x": int(x), "y": int(y), "button": button_name}
            )

        if button == mouse.Button.left:
            self._drag_start = None
            self._is_dragging = False

    def _on_move(self, x: int, y: int) -> None:
        if self._drag_start is None:
            return
        dx = abs(int(x) - self._drag_start[0])
        dy = abs(int(y) - self._drag_start[1])
        if not self._is_dragging and (dx > 4 or dy > 4):
            self._is_dragging = True
            self._write_event(
                "mouse_drag_start",
                {
                    "x": self._drag_start[0],
                    "y": self._drag_start[1],
                },
            )
        if self._is_dragging:
            self._write_event("mouse_drag_move", {"x": int(x), "y": int(y)})

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        direction = "up" if dy > 0 else "down"
        self._write_event(
            "mouse_scroll",
            {
                "x": int(x),
                "y": int(y),
                "dx": int(dx),
                "dy": int(dy),
                "direction": direction,
            },
        )
