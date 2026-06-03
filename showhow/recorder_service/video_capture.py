from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _resolve_ffmpeg_bin(default: str) -> str:
    """On Windows, prefer an ffmpeg binary that supports gdigrab over the default."""
    if default and default != "ffmpeg":
        return default
    if platform.system() != "Windows":
        resolved_default = default or "ffmpeg"
        if shutil.which(resolved_default):
            return resolved_default
        try:
            import imageio_ffmpeg

            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled:
                logger.warning("Selected bundled imageio-ffmpeg binary: %s", bundled)
                return str(bundled)
        except Exception:
            pass
        return default or "ffmpeg"

    # Walk all ffmpeg binaries in PATH and pick the first one with gdigrab support.
    candidates: list[str] = []
    seen: set[str] = set()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(entry) / "ffmpeg.exe"
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            candidates.append(str(candidate))

    for bin_path in candidates:
        try:
            result = subprocess.run(
                [bin_path, "-devices", "-v", "quiet"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            combined = result.stdout + result.stderr
            if "gdigrab" in combined.lower():
                logger.warning("Selected ffmpeg with gdigrab support: %s", bin_path)
                return bin_path
        except Exception:
            continue

    return default or "ffmpeg"


@dataclass
class VideoCaptureConfig:
    ffmpeg_bin: str = field(
        default_factory=lambda: _resolve_ffmpeg_bin(os.getenv("SHOWHOW_FFMPEG_BIN", ""))
    )
    framerate: int = int(os.getenv("SHOWHOW_VIDEO_FPS", "30"))
    bitrate: str = os.getenv("SHOWHOW_VIDEO_BITRATE", "10000k")
    screen_number: int = int(os.getenv("SHOWHOW_SCREEN_NUMBER", "0"))
    input_format: str = os.getenv("SHOWHOW_VIDEO_INPUT_FORMAT", "").strip().lower()


class FFmpegVideoRecorder:
    def __init__(
        self, output_path: Path, config: Optional[VideoCaptureConfig] = None
    ) -> None:
        self.output_path = output_path
        self.config = config or VideoCaptureConfig()
        self._proc: Optional[subprocess.Popen] = None

    @property
    def is_recording(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        args = self._build_capture_args()
        logger.warning("Starting ffmpeg: %s", " ".join(str(a) for a in args))

        is_windows = platform.system() == "Windows"
        kwargs: dict = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if is_windows:
            # Isolate ffmpeg in its own process group so CTRL_BREAK_EVENT only reaches it.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        self._proc = subprocess.Popen(args, **kwargs)
        # Give ffmpeg a brief startup window to fail fast if device binding is invalid.
        time.sleep(0.5)
        if self._proc.poll() is not None:
            stderr_output = ""
            try:
                stderr_output = (
                    self._proc.stderr.read().decode("utf-8", errors="replace")
                    if self._proc.stderr
                    else ""
                )
            except Exception:
                pass
            raise RuntimeError(
                f"ffmpeg exited immediately on startup. stderr: {stderr_output}"
            )

    def stop(self) -> None:
        if not self._proc:
            return
        proc = self._proc
        self._proc = None

        if proc.poll() is not None:
            return

        # Use communicate() to send 'q' AND drain the stderr pipe simultaneously.
        # Plain proc.wait() after stdin.write() can deadlock when ffmpeg fills the
        # stderr pipe buffer (64 KB) before exiting – communicate() avoids that.
        try:
            proc.communicate(input=b"q", timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.communicate(timeout=3)
            except Exception:
                pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        self._remux_if_truncated(self.output_path)

    def _remux_if_truncated(self, path: Path) -> None:
        """Re-mux the recorded file to fix container truncation from non-graceful shutdown."""
        if not path.exists() or path.stat().st_size == 0:
            return
        tmp = path.with_suffix(".remux.mkv")
        try:
            subprocess.run(
                [self.config.ffmpeg_bin, "-y", "-i", str(path), "-c", "copy", str(tmp)],
                timeout=120,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(path)
                logger.warning("Remuxed video to fix container: %s", path)
        except Exception as exc:
            logger.warning("Remux failed (non-fatal): %s", exc)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def _detect_avfoundation_device_index(self) -> int:
        args = [
            self.config.ffmpeg_bin,
            "-f",
            "avfoundation",
            "-list_devices",
            "true",
            "-i",
            "",
        ]
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

        matches = re.findall(
            r"\[(\d+)\]\s+Capture screen\s+(\d+)", combined, flags=re.IGNORECASE
        )
        if not matches:
            # Some ffmpeg builds label the device as "Screen <n>" instead of
            # "Capture screen <n>"; keep the parser tolerant across macOS builds.
            matches = re.findall(
                r"\[(\d+)\]\s+Screen\s+(\d+)", combined, flags=re.IGNORECASE
            )
        if not matches:
            raise RuntimeError("No AVFoundation screen devices found via ffmpeg")

        mapping: dict[int, int] = {}
        for device_idx, screen_num in matches:
            mapping[int(screen_num)] = int(device_idx)

        if self.config.screen_number in mapping:
            return mapping[self.config.screen_number]

        # Fallback to the first available screen device.
        first_screen = sorted(mapping.keys())[0]
        return mapping[first_screen]

    def _build_capture_args(self) -> list[str]:
        input_format = self.config.input_format or self._default_input_format()
        if input_format == "avfoundation":
            return self._build_macos_args()
        if input_format == "gdigrab":
            return self._build_windows_args()
        raise RuntimeError(
            f"Unsupported ffmpeg input format: {input_format}. "
            "Set SHOWHOW_VIDEO_INPUT_FORMAT=avfoundation|gdigrab."
        )

    def _default_input_format(self) -> str:
        system = platform.system()
        if system == "Darwin":
            return "avfoundation"
        if system == "Windows":
            return "gdigrab"
        return "unsupported"

    def _build_encoder_args(self) -> list[str]:
        common = [
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-r",
            str(self.config.framerate),
            "-fps_mode",
            "cfr",
        ]
        if platform.system() == "Darwin" and self._encoder_available(
            "h264_videotoolbox"
        ):
            return [
                *common,
                "-c:v",
                "h264_videotoolbox",
                "-allow_sw",
                "1",
                "-b:v",
                self.config.bitrate,
                "-pix_fmt",
                "yuv420p",
                str(self.output_path),
            ]
        return [
            *common,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-b:v",
            self.config.bitrate,
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]

    def _encoder_available(self, encoder_name: str) -> bool:
        try:
            result = subprocess.run(
                [self.config.ffmpeg_bin, "-hide_banner", "-encoders"],
                check=False,
                capture_output=True,
                text=True,
                timeout=8,
            )
        except Exception:
            return False
        combined = f"{result.stdout}\n{result.stderr}"
        return encoder_name in combined

    def _build_macos_args(self) -> list[str]:
        device_idx = self._detect_avfoundation_device_index()
        return [
            self.config.ffmpeg_bin,
            "-f",
            "avfoundation",
            "-pixel_format",
            "uyvy422",
            "-framerate",
            str(self.config.framerate),
            "-capture_cursor",
            "0",
            "-use_wallclock_as_timestamps",
            "1",
            "-thread_queue_size",
            "2048",
            "-probesize",
            "100M",
            "-i",
            f"{device_idx}:none",
            *self._build_encoder_args(),
        ]

    def _build_windows_args(self) -> list[str]:
        return [
            self.config.ffmpeg_bin,
            "-f",
            "gdigrab",
            "-framerate",
            str(self.config.framerate),
            "-draw_mouse",
            "1",
            "-i",
            "desktop",
            *self._build_encoder_args(),
        ]
