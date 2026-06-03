from __future__ import annotations

import ctypes
import platform
import subprocess
from ctypes import wintypes
from pathlib import Path
from typing import Any, Optional, Protocol


class ActiveContextProvider(Protocol):
    def is_input_capture_trusted(self) -> bool: ...

    def capture_active_context(
        self, *, input_capture_enabled: bool
    ) -> dict[str, Any]: ...


def _empty_context() -> dict[str, Any]:
    return {
        "app_name": None,
        "bundle_id": None,
        "pid": None,
        "window_title": None,
        "url": None,
        "ui": None,
    }


class NullContextProvider:
    def is_input_capture_trusted(self) -> bool:
        return True

    def capture_active_context(self, *, input_capture_enabled: bool) -> dict[str, Any]:
        return _empty_context()


class MacOSContextProvider:
    def __init__(self) -> None:
        try:
            from AppKit import NSWorkspace
        except Exception:  # pragma: no cover - optional dependency
            NSWorkspace = None
        self._NSWorkspace = NSWorkspace

        try:
            from ApplicationServices import (
                AXIsProcessTrusted,
                AXUIElementCopyAttributeValue,
                AXUIElementCreateApplication,
                kAXDescriptionAttribute,
                kAXFocusedUIElementAttribute,
                kAXFocusedWindowAttribute,
                kAXRoleAttribute,
                kAXSubroleAttribute,
                kAXTitleAttribute,
                kAXValueAttribute,
            )
        except Exception:  # pragma: no cover - optional dependency
            AXIsProcessTrusted = None
            AXUIElementCopyAttributeValue = None
            AXUIElementCreateApplication = None
            kAXDescriptionAttribute = None
            kAXFocusedUIElementAttribute = None
            kAXFocusedWindowAttribute = None
            kAXRoleAttribute = None
            kAXSubroleAttribute = None
            kAXTitleAttribute = None
            kAXValueAttribute = None
        self._AXIsProcessTrusted = AXIsProcessTrusted
        self._AXUIElementCopyAttributeValue = AXUIElementCopyAttributeValue
        self._AXUIElementCreateApplication = AXUIElementCreateApplication
        self._kAXDescriptionAttribute = kAXDescriptionAttribute
        self._kAXFocusedUIElementAttribute = kAXFocusedUIElementAttribute
        self._kAXFocusedWindowAttribute = kAXFocusedWindowAttribute
        self._kAXRoleAttribute = kAXRoleAttribute
        self._kAXSubroleAttribute = kAXSubroleAttribute
        self._kAXTitleAttribute = kAXTitleAttribute
        self._kAXValueAttribute = kAXValueAttribute

        try:
            from Quartz import (  # type: ignore
                CGWindowListCopyWindowInfo,
                kCGNullWindowID,
                kCGWindowListOptionOnScreenOnly,
            )
        except Exception:  # pragma: no cover - optional dependency
            CGWindowListCopyWindowInfo = None
            kCGNullWindowID = None
            kCGWindowListOptionOnScreenOnly = None
        self._CGWindowListCopyWindowInfo = CGWindowListCopyWindowInfo
        self._kCGNullWindowID = kCGNullWindowID
        self._kCGWindowListOptionOnScreenOnly = kCGWindowListOptionOnScreenOnly

    def is_input_capture_trusted(self) -> bool:
        if self._AXIsProcessTrusted is None:
            return True
        try:
            return bool(self._AXIsProcessTrusted())
        except Exception:
            return True

    def capture_active_context(self, *, input_capture_enabled: bool) -> dict[str, Any]:
        context = _empty_context()
        if self._NSWorkspace is None:
            return context

        try:
            app = self._NSWorkspace.sharedWorkspace().frontmostApplication()
            if not app:
                return context
            bundle_id = str(app.bundleIdentifier() or "")
            pid = int(app.processIdentifier())
            context["app_name"] = str(app.localizedName() or "UnknownApp")
            context["bundle_id"] = bundle_id
            context["pid"] = pid
            context["window_title"] = self._frontmost_window_title(pid)
            context["url"] = self._frontmost_browser_url(bundle_id)
            context["ui"] = self._focused_ui_semantics(
                pid, input_capture_enabled=input_capture_enabled
            )
            return context
        except Exception:
            return context

    def _frontmost_window_title(self, pid: Optional[int]) -> Optional[str]:
        if not pid:
            return None
        if (
            self._CGWindowListCopyWindowInfo is None
            or self._kCGWindowListOptionOnScreenOnly is None
            or self._kCGNullWindowID is None
        ):
            return None
        try:
            infos = self._CGWindowListCopyWindowInfo(
                self._kCGWindowListOptionOnScreenOnly, self._kCGNullWindowID
            )
            if not infos:
                return None
            for info in infos:
                owner_pid = int(info.get("kCGWindowOwnerPID", -1))
                if owner_pid != int(pid):
                    continue
                layer = int(info.get("kCGWindowLayer", 1))
                if layer != 0:
                    continue
                title = str(info.get("kCGWindowName", "") or "").strip()
                if title:
                    return title
        except Exception:
            return None
        return None

    def _frontmost_browser_url(self, bundle_id: Optional[str]) -> Optional[str]:
        if not bundle_id:
            return None

        chrome_like = {
            "com.google.Chrome": "Google Chrome",
            "com.google.Chrome.canary": "Google Chrome Canary",
            "com.brave.Browser": "Brave Browser",
            "com.microsoft.edgemac": "Microsoft Edge",
            "company.thebrowser.Browser": "Arc",
        }
        if bundle_id in chrome_like:
            script = f'tell application "{chrome_like[bundle_id]}" to get URL of active tab of front window'
            return self._run_osascript(script)

        if bundle_id == "com.apple.Safari":
            return self._run_osascript(
                'tell application "Safari" to get URL of front document'
            )
        return None

    def _run_osascript(self, script: str) -> Optional[str]:
        try:
            proc = subprocess.run(
                ["osascript", "-e", script],
                check=False,
                capture_output=True,
                text=True,
                timeout=0.25,
            )
            if proc.returncode != 0:
                return None
            out = (proc.stdout or "").strip()
            return out or None
        except Exception:
            return None

    def _focused_ui_semantics(
        self, pid: Optional[int], *, input_capture_enabled: bool
    ) -> Optional[dict[str, Any]]:
        if not pid:
            return None
        if (
            self._AXUIElementCreateApplication is None
            or self._AXUIElementCopyAttributeValue is None
        ):
            return None
        if not input_capture_enabled:
            return None
        if (
            self._kAXFocusedUIElementAttribute is None
            or self._kAXFocusedWindowAttribute is None
            or self._kAXRoleAttribute is None
            or self._kAXSubroleAttribute is None
            or self._kAXTitleAttribute is None
            or self._kAXDescriptionAttribute is None
            or self._kAXValueAttribute is None
        ):
            return None
        try:
            app_el = self._AXUIElementCreateApplication(int(pid))
            err, focused = self._AXUIElementCopyAttributeValue(
                app_el, self._kAXFocusedUIElementAttribute, None
            )
            if err != 0 or focused is None:
                err, focused = self._AXUIElementCopyAttributeValue(
                    app_el, self._kAXFocusedWindowAttribute, None
                )
                if err != 0 or focused is None:
                    return None

            role = self._ax_attr(focused, self._kAXRoleAttribute)
            subrole = self._ax_attr(focused, self._kAXSubroleAttribute)
            title = self._ax_attr(focused, self._kAXTitleAttribute)
            desc = self._ax_attr(focused, self._kAXDescriptionAttribute)
            value = self._ax_attr(focused, self._kAXValueAttribute)

            semantics = {
                "role": role,
                "subrole": subrole,
                "title": title,
                "description": desc,
                "has_value": value is not None and str(value).strip() != "",
                "value_preview": self._truncate(str(value), 80)
                if value is not None
                else None,
            }
            if (
                semantics["role"] is None
                and semantics["title"] is None
                and semantics["description"] is None
            ):
                return None
            return semantics
        except Exception:
            return None

    def _ax_attr(self, element: Any, attr: Any) -> Optional[str]:
        if self._AXUIElementCopyAttributeValue is None:
            return None
        try:
            err, value = self._AXUIElementCopyAttributeValue(element, attr, None)
            if err != 0 or value is None:
                return None
            return self._truncate(str(value), 200)
        except Exception:
            return None

    def _truncate(self, text: str, max_len: int) -> str:
        t = text.strip()
        if len(t) <= max_len:
            return t
        return t[: max_len - 3] + "..."


class WindowsContextProvider:
    def __init__(self) -> None:
        self._user32 = (
            getattr(ctypes, "windll", None).user32
            if hasattr(ctypes, "windll")
            else None
        )
        self._kernel32 = (
            getattr(ctypes, "windll", None).kernel32
            if hasattr(ctypes, "windll")
            else None
        )

    def is_input_capture_trusted(self) -> bool:
        return True

    def capture_active_context(self, *, input_capture_enabled: bool) -> dict[str, Any]:
        context = _empty_context()
        if self._user32 is None:
            return context

        hwnd = self._user32.GetForegroundWindow()
        if not hwnd:
            return context

        pid = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        pid_int = int(pid.value)

        window_title = self._window_text(hwnd)
        exe_path = self._process_exe(pid_int)
        app_name = Path(exe_path).name if exe_path else None

        context["app_name"] = app_name
        context["bundle_id"] = app_name
        context["pid"] = pid_int if pid_int > 0 else None
        context["window_title"] = window_title
        context["url"] = self._url_from_title(window_title, app_name)
        context["ui"] = self._focused_ui_semantics()
        context["process_exe"] = exe_path
        return context

    def _window_text(self, hwnd: int) -> Optional[str]:
        try:
            length = self._user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return None
            buffer = ctypes.create_unicode_buffer(length + 1)
            self._user32.GetWindowTextW(hwnd, buffer, length + 1)
            text = buffer.value.strip()
            return text or None
        except Exception:
            return None

    def _process_exe(self, pid: int) -> Optional[str]:
        if pid <= 0 or self._kernel32 is None:
            return None
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = self._kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not handle:
            return None
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            ok = self._kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)
            )
            if not ok:
                return None
            value = buffer.value.strip()
            return value or None
        except Exception:
            return None
        finally:
            try:
                self._kernel32.CloseHandle(handle)
            except Exception:
                pass

    def _focused_ui_semantics(self) -> Optional[dict[str, Any]]:
        try:
            import uiautomation as auto  # type: ignore
        except Exception:
            return None

        try:
            ctrl = auto.GetFocusedControl()
            if not ctrl:
                return None
            name = (getattr(ctrl, "Name", None) or "").strip() or None
            role = (getattr(ctrl, "ControlTypeName", None) or "").strip() or None
            value = None
            try:
                value = ctrl.GetValuePattern().Value
            except Exception:
                value = None
            semantics = {
                "role": role,
                "subrole": None,
                "title": name,
                "description": None,
                "has_value": value is not None and str(value).strip() != "",
                "value_preview": self._truncate(str(value), 80)
                if value is not None
                else None,
            }
            if semantics["role"] is None and semantics["title"] is None:
                return None
            return semantics
        except Exception:
            return None

    def _url_from_title(
        self, title: Optional[str], app_name: Optional[str]
    ) -> Optional[str]:
        if not title or not app_name:
            return None
        lower = app_name.lower()
        browser_markers = ("chrome", "msedge", "firefox", "brave", "opera")
        if not any(marker in lower for marker in browser_markers):
            return None
        # URL extraction on Windows is intentionally conservative and best-effort.
        if " - " in title:
            maybe = title.split(" - ", 1)[0].strip()
            if "." in maybe and " " not in maybe:
                return maybe
        return None

    def _truncate(self, text: str, max_len: int) -> str:
        t = text.strip()
        if len(t) <= max_len:
            return t
        return t[: max_len - 3] + "..."


def create_context_provider() -> ActiveContextProvider:
    system = platform.system()
    if system == "Darwin":
        return MacOSContextProvider()
    if system == "Windows":
        return WindowsContextProvider()
    return NullContextProvider()
