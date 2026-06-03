from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .recorder_client import RecorderHTTPClient


class RecorderProcessManager:
    def __init__(
        self,
        client: RecorderHTTPClient,
        command: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,
        startup_timeout_sec: float = 12.0,
        poll_interval_sec: float = 0.3,
    ) -> None:
        self.client = client
        self.command = command or [
            sys.executable,
            "-m",
            "showhow.recorder_service.main",
        ]
        self.env = env or os.environ.copy()
        self.startup_timeout_sec = startup_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self._proc: Optional[subprocess.Popen] = None
        self._log_path = Path(tempfile.gettempdir()) / "showhow_recorder_child.log"

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.is_running():
            return

        log_file = self._log_path.open("a", encoding="utf-8")
        self._proc = subprocess.Popen(
            self.command,
            env=self.env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.time() + self.startup_timeout_sec
        last_error: Optional[Exception] = None
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("Recorder process exited during startup")
            try:
                health = self.client.health()
                if health.get("status") == "ok":
                    return
            except Exception as exc:  # noqa: PERF203
                last_error = exc
            time.sleep(self.poll_interval_sec)

        self.stop(force=True)
        raise RuntimeError(f"Recorder failed health check before timeout: {last_error}")

    def stop(self, force: bool = False) -> None:
        if not self._proc:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if force:
                    self._proc.kill()
                    self._proc.wait(timeout=2)
        self._proc = None

    def restart(self) -> None:
        self.stop(force=True)
        self.start()
