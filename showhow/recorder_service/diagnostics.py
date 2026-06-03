from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def run_diagnostics() -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": sys.platform,
        "python_executable": sys.executable,
        "ffmpeg_bin": _resolve_ffmpeg_bin(os.getenv("SHOWHOW_FFMPEG_BIN", "ffmpeg")),
        "ffmpeg_bin_configured": os.getenv("SHOWHOW_FFMPEG_BIN", "ffmpeg"),
        "ffmpeg_available": False,
        "ffmpeg_screen_backend": None,
        "ffmpeg_screen_backend_available": None,
        "accessibility_trusted": None,
        "macos_pyobjc_available": None,
        "notes": [],
    }

    ffmpeg_bin = report["ffmpeg_bin"]
    ffmpeg_path = shutil.which(str(ffmpeg_bin))
    report["ffmpeg_available"] = ffmpeg_path is not None
    report["ffmpeg_path"] = ffmpeg_path

    if sys.platform == "darwin":
        report["ffmpeg_screen_backend"] = "avfoundation"
        report["ffmpeg_screen_backend_available"] = _ffmpeg_has_device(
            ffmpeg_bin, "avfoundation"
        )
        if report["ffmpeg_available"] and not report["ffmpeg_screen_backend_available"]:
            report["notes"].append(
                "ffmpeg is available but does not list the avfoundation input device. "
                "Install a full ffmpeg build, for example with Homebrew."
            )
        try:
            from AppKit import NSWorkspace  # noqa: F401
            from ApplicationServices import AXIsProcessTrusted
            from Quartz import CGWindowListCopyWindowInfo  # noqa: F401

            report["macos_pyobjc_available"] = True
            trusted = bool(AXIsProcessTrusted())
            report["accessibility_trusted"] = trusted
            if not trusted:
                report["notes"].append(
                    "Accessibility trust is OFF for the current Python process. "
                    "Grant Accessibility to your terminal app and Python executable, then restart terminal."
                )
        except Exception as exc:
            report["macos_pyobjc_available"] = False
            report["accessibility_trusted"] = None
            report["notes"].append(
                f"Unable to load macOS context dependencies or query accessibility trust: {exc}"
            )
        report["notes"].append(
            "For macOS recording, grant Screen Recording, Accessibility, and Input Monitoring "
            "to your terminal app/Python executable in System Settings, then restart terminal."
        )
    elif platform.system() == "Windows":
        report["ffmpeg_screen_backend"] = "gdigrab"
        report["ffmpeg_screen_backend_available"] = _ffmpeg_has_device(
            ffmpeg_bin, "gdigrab"
        )
        if report["ffmpeg_available"] and not report["ffmpeg_screen_backend_available"]:
            report["notes"].append(
                "ffmpeg is available but does not list the gdigrab input device. "
                "Install a Windows ffmpeg build with gdigrab support."
            )

    if not report["ffmpeg_available"]:
        report["notes"].append(
            "ffmpeg is not found. Install ffmpeg or set SHOWHOW_FFMPEG_BIN to an absolute binary path."
        )

    report["record_root"] = str(
        Path(
            os.getenv(
                "SHOWHOW_RECORD_ROOT", str(Path.home() / "Downloads" / "record_save")
            )
        ).expanduser()
    )
    return report


def _ffmpeg_has_device(ffmpeg_bin: str, device_name: str) -> bool | None:
    if not shutil.which(str(ffmpeg_bin)):
        return None
    try:
        result = subprocess.run(
            [str(ffmpeg_bin), "-hide_banner", "-devices"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return device_name.lower() in combined


def _resolve_ffmpeg_bin(configured: str) -> str:
    if configured and shutil.which(configured):
        return configured
    if configured and configured != "ffmpeg":
        return configured
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            return str(bundled)
    except Exception:
        pass
    return configured or "ffmpeg"
