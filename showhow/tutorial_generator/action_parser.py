from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from showhow.tutorial_generator.types import ParseResult

logger = logging.getLogger(__name__)

MODIFIER_KEYS = {
    "shift",
    "shift_l",
    "shift_r",
    "ctrl",
    "ctrl_l",
    "ctrl_r",
    "cmd",
    "cmd_l",
    "cmd_r",
    "alt",
    "alt_l",
    "alt_r",
}
TYPEABLE_SPECIAL = {
    "space": " ",
    "tab": "\t",
}


def _showhow_ui_ports() -> set[str]:
    """Return the set of ports that belong to the ShowHow web UI."""
    port = os.getenv("SHOWHOW_WEB_PORT", "18090").strip()
    return {port, "18090"}


_BROWSER_EXE_MARKERS = ("chrome", "msedge", "firefox", "brave", "opera", "edge")


def _is_showhow_ui_event(ev: dict[str, Any]) -> bool:
    """Return True if this event was captured while the ShowHow web UI was active.

    Two detection strategies:
    - URL-based (macOS / platforms where URL is captured): check for `:18090` in url.
    - Title-based (Windows fallback): check for "showhow" in window title inside a browser.
    """
    # --- URL-based (reliable on macOS) ---
    url: str = str(ev.get("url") or "").lower()
    if url:
        for port in _showhow_ui_ports():
            if f":{port}" in url:
                return True

    # --- Title-based (Windows: URL not captured for localhost, but title is) ---
    window_title = str(ev.get("window_title") or "").lower()
    app_name = str(ev.get("window") or "").lower()
    if window_title and "showhow" in window_title:
        if any(b in app_name for b in _BROWSER_EXE_MARKERS):
            return True

    return False


@dataclass
class CompactAction:
    t_start: float
    t_end: float
    action: str
    software: str
    points: list[dict[str, int]]
    context: dict[str, Any]


def build_compact_trace_from_events(data_path: Path) -> ParseResult:
    events_path = data_path / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"events.jsonl not found in session path: {data_path}")

    events, last_showhow_t = _load_events(events_path)
    compact = _compact_events(events)
    compact = _trim_pre_showhow_stop(compact, last_showhow_t)
    trace_path = data_path / "trace_compact.json"
    _write_trace(compact, trace_path)
    return ParseResult(trace_json_path=trace_path)


def _load_events(events_path: Path) -> tuple[list[dict[str, Any]], Optional[float]]:
    """Return (filtered events, timestamp of the last ShowHow UI event or None)."""
    rows: list[dict[str, Any]] = []
    skipped = 0
    last_showhow_t: Optional[float] = None
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            elapsed_ms = float(ev.get("elapsed_ms", 0.0))
            t = max(0.0, elapsed_ms / 1000.0)
            if _is_showhow_ui_event(ev):
                skipped += 1
                if last_showhow_t is None or t > last_showhow_t:
                    last_showhow_t = t
                continue
            ev["t"] = t
            rows.append(ev)
    rows.sort(key=lambda e: e.get("t", 0.0))
    if skipped:
        logger.info("Parser: filtered %d ShowHow UI events from event log", skipped)
    return rows, last_showhow_t


def _compact_events(events: list[dict[str, Any]]) -> list[CompactAction]:
    actions: list[CompactAction] = []
    typing_buffer: list[str] = []
    typing_start: Optional[float] = None
    typing_end: Optional[float] = None
    typing_window = "UnknownApp"
    typing_context: dict[str, Any] = {}

    drag_start: Optional[dict[str, int]] = None
    drag_last: Optional[dict[str, int]] = None
    drag_t_start: Optional[float] = None
    drag_window = "UnknownApp"
    drag_context: dict[str, Any] = {}

    pending_click: Optional[tuple[float, str, dict[str, int], str, dict[str, Any]]] = (
        None
    )

    scroll_dir: Optional[str] = None
    scroll_count = 0
    scroll_t_start: Optional[float] = None
    scroll_t_end: Optional[float] = None
    scroll_point: Optional[dict[str, int]] = None
    scroll_window = "UnknownApp"
    scroll_context: dict[str, Any] = {}

    def flush_typing() -> None:
        nonlocal typing_buffer, typing_start, typing_end, typing_window, typing_context
        if not typing_buffer:
            return
        text = "".join(typing_buffer)
        actions.append(
            CompactAction(
                t_start=typing_start or 0.0,
                t_end=typing_end or (typing_start or 0.0),
                action=f"Type: {text}",
                software=typing_window,
                points=[],
                context=dict(typing_context),
            )
        )
        typing_buffer = []
        typing_start = None
        typing_end = None
        typing_context = {}

    def flush_scroll() -> None:
        nonlocal \
            scroll_dir, \
            scroll_count, \
            scroll_t_start, \
            scroll_t_end, \
            scroll_point, \
            scroll_window, \
            scroll_context
        if not scroll_dir or scroll_count <= 0:
            return
        pt = scroll_point or {"x": 0, "y": 0}
        actions.append(
            CompactAction(
                t_start=scroll_t_start or 0.0,
                t_end=scroll_t_end or (scroll_t_start or 0.0),
                action=f"Scroll{scroll_dir.capitalize()} x{scroll_count} at ({pt['x']}, {pt['y']})",
                software=scroll_window,
                points=[pt],
                context=dict(scroll_context),
            )
        )
        scroll_dir = None
        scroll_count = 0
        scroll_t_start = None
        scroll_t_end = None
        scroll_point = None
        scroll_context = {}

    def flush_pending_click() -> None:
        nonlocal pending_click
        if not pending_click:
            return
        t, button, pt, win, ctx = pending_click
        actions.append(
            CompactAction(
                t_start=t,
                t_end=t,
                action=f"{button.capitalize()}Click at ({pt['x']}, {pt['y']})",
                software=win,
                points=[pt],
                context=dict(ctx),
            )
        )
        pending_click = None

    for ev in events:
        ev_type = str(ev.get("event_type", ""))
        payload = ev.get("payload") or {}
        t = float(ev.get("t", 0.0))
        win = str(ev.get("window") or "UnknownApp")
        ctx = _context_from_event(ev)

        # Typing timeout flush.
        if typing_buffer and typing_end is not None and (t - typing_end) > 1.2:
            flush_typing()

        # Scroll boundary flush.
        if scroll_dir and ev_type != "mouse_scroll":
            flush_scroll()

        if ev_type == "mouse_drag_start":
            flush_typing()
            flush_pending_click()
            pt = _point(payload)
            if pt:
                drag_start = pt
                drag_last = pt
                drag_t_start = t
                drag_window = win
                drag_context = ctx
            continue

        if ev_type == "mouse_drag_move":
            pt = _point(payload)
            if pt:
                drag_last = pt
            continue

        if ev_type == "mouse_drag_end":
            pt = _point(payload)
            if pt:
                drag_last = pt
            if drag_start and drag_last and drag_t_start is not None:
                actions.append(
                    CompactAction(
                        t_start=drag_t_start,
                        t_end=t,
                        action=f"Drag from ({drag_start['x']}, {drag_start['y']}) to ({drag_last['x']}, {drag_last['y']})",
                        software=drag_window,
                        points=[drag_start, drag_last],
                        context=dict(drag_context),
                    )
                )
            drag_start = None
            drag_last = None
            drag_t_start = None
            drag_context = {}
            continue

        if ev_type == "mouse_down":
            flush_typing()
            flush_scroll()
            flush_pending_click()
            button = str(payload.get("button", "left"))
            pt = _point(payload)
            if pt:
                pending_click = (t, button, pt, win, ctx)
            continue

        if ev_type == "mouse_up":
            # keep click when mouse_down+mouse_up pair appears without drag
            continue

        if ev_type == "mouse_scroll":
            flush_typing()
            direction = str(payload.get("direction", "down")).lower()
            pt = _point(payload)
            if scroll_dir is None:
                scroll_dir = direction
                scroll_count = 1
                scroll_t_start = t
                scroll_t_end = t
                scroll_point = pt
                scroll_window = win
                scroll_context = ctx
            elif scroll_dir == direction:
                scroll_count += 1
                scroll_t_end = t
                if pt:
                    scroll_point = pt
                if not scroll_context:
                    scroll_context = ctx
            else:
                flush_scroll()
                scroll_dir = direction
                scroll_count = 1
                scroll_t_start = t
                scroll_t_end = t
                scroll_point = pt
                scroll_window = win
                scroll_context = ctx
            continue

        if ev_type == "key_press":
            key_raw = str(payload.get("key", "")).strip()
            key = key_raw.lower()
            modifiers = {str(m).lower() for m in (payload.get("modifiers") or [])}

            if key in MODIFIER_KEYS:
                continue

            if key == "enter":
                flush_typing()
                flush_pending_click()
                actions.append(
                    CompactAction(
                        t_start=t,
                        t_end=t,
                        action="Press Enter",
                        software=win,
                        points=[],
                        context=ctx,
                    )
                )
                continue

            if _is_typeable_key(key):
                flush_pending_click()
                char = _normalize_key_to_char(key_raw)
                if char is None:
                    continue
                if typing_start is None:
                    typing_start = t
                    typing_window = win
                    typing_context = ctx
                typing_end = t
                typing_buffer.append(char)
                continue

            # Non-typeable key press becomes explicit hotkey/action.
            flush_typing()
            flush_pending_click()
            mods = sorted(
                m
                for m in modifiers
                if m not in MODIFIER_KEYS or m in {"cmd", "ctrl", "alt", "shift"}
            )
            hotkey = (
                "+".join([m.upper() for m in mods] + [key_raw.upper()])
                if mods
                else key_raw.upper()
            )
            actions.append(
                CompactAction(
                    t_start=t,
                    t_end=t,
                    action=f"Hotkey: {hotkey}",
                    software=win,
                    points=[],
                    context=ctx,
                )
            )
            continue

        if ev_type == "key_release":
            # Ignore individual releases to avoid sparse sequences.
            continue

        # Unknown events flush current buffers and ignore as dummy noise.
        flush_typing()
        flush_scroll()
        flush_pending_click()

    flush_typing()
    flush_scroll()
    flush_pending_click()

    # Final denoise pass.
    actions = _drop_redundant(actions)
    actions = _suppress_scroll_jitter(actions)
    actions = _semantic_text_edit_cleanup(actions)
    actions = _merge_hotkey_confirm_pairs(actions)
    return actions


_PRE_SHOWHOW_TRIM_WINDOW = 6.0  # seconds before last ShowHow UI event to inspect


def _trim_pre_showhow_stop(
    actions: list[CompactAction], last_showhow_t: Optional[float]
) -> list[CompactAction]:
    """Remove the trailing action(s) that were switching back to ShowHow to stop recording.

    Strategy: the last ShowHow UI event (filtered out of the trace) marks when the user
    arrived at the ShowHow control page.  Any compact action whose t_end falls within
    _PRE_SHOWHOW_TRIM_WINDOW seconds before that timestamp is a candidate for trimming —
    but only the LAST contiguous run of such candidates is removed (so we don't touch
    real tutorial steps earlier in the recording).
    """
    if last_showhow_t is None or not actions:
        return actions

    cutoff = last_showhow_t - _PRE_SHOWHOW_TRIM_WINDOW
    trim_from = len(actions)
    for i in range(len(actions) - 1, -1, -1):
        act = actions[i]
        if act.t_end < cutoff:
            break
        # Only trim actions that look like "switching window / navigation noise",
        # not meaningful tutorial steps (type, drag, multi-step hotkeys, etc.)
        if _is_window_switch_action(act):
            trim_from = i
        else:
            break  # stop at the first real tutorial action from the end

    if trim_from < len(actions):
        logger.info(
            "Parser: trimmed %d trailing window-switch action(s) before ShowHow stop (t=%.2f)",
            len(actions) - trim_from,
            last_showhow_t,
        )
    return actions[:trim_from]


def _is_window_switch_action(act: CompactAction) -> bool:
    """True for actions that are likely transitioning windows to reach ShowHow."""
    action = act.action
    # Alt+Tab / Windows+Tab / Windows+number
    if action.startswith("Hotkey: "):
        key = action.removeprefix("Hotkey: ").strip()
        return key in {"ALT+TAB", "WINDOWS+TAB"} or (
            key.startswith("WINDOWS+") and len(key) <= 10
        )
    # Single bare click (no meaningful content) — might be taskbar click
    if action.startswith("LClick at") or action.startswith("RClick at"):
        return True
    return False


def _drop_redundant(actions: list[CompactAction]) -> list[CompactAction]:
    cleaned: list[CompactAction] = []
    for act in actions:
        if act.action.strip() in {"Type: ", "Hotkey: "}:
            continue
        if cleaned:
            prev = cleaned[-1]
            if act.action == prev.action and abs(act.t_start - prev.t_end) < 0.05:
                # Merge near-duplicate consecutive events.
                prev.t_end = max(prev.t_end, act.t_end)
                continue
        cleaned.append(act)
    return cleaned


def _suppress_scroll_jitter(actions: list[CompactAction]) -> list[CompactAction]:
    out: list[CompactAction] = []
    i = 0
    while i < len(actions):
        if not _is_scroll_action(actions[i].action):
            out.append(actions[i])
            i += 1
            continue
        if (actions[i].t_end - actions[i].t_start) > 1.5:
            # Keep long continuous scrolls as intentional actions.
            out.append(actions[i])
            i += 1
            continue

        j = i + 1
        while j < len(actions):
            if not _is_scroll_action(actions[j].action):
                break
            if actions[j].software != actions[i].software:
                break
            if (actions[j].t_end - actions[j].t_start) > 1.5:
                break
            if _point_distance(actions[i], actions[j]) > 8:
                break
            if (actions[j].t_start - actions[j - 1].t_end) > 0.4:
                break
            j += 1

        cluster = actions[i:j]
        if len(cluster) >= 4:
            span = cluster[-1].t_end - cluster[0].t_start
            changes = 0
            prev_dir: Optional[str] = None
            net = 0
            for act in cluster:
                direction, count, point = _parse_scroll_action(act.action)
                if direction is None or point is None:
                    continue
                if prev_dir is not None and prev_dir != direction:
                    changes += 1
                prev_dir = direction
                sign = 1 if direction == "down" else -1
                net += sign * count

            if span <= 1.2 and changes >= 2:
                if abs(net) <= 2:
                    i = j
                    continue
                direction = "Down" if net > 0 else "Up"
                point = (
                    cluster[-1].points[-1] if cluster[-1].points else {"x": 0, "y": 0}
                )
                out.append(
                    CompactAction(
                        t_start=cluster[0].t_start,
                        t_end=cluster[-1].t_end,
                        action=f"Scroll{direction} x{abs(net)} at ({point['x']}, {point['y']})",
                        software=cluster[0].software,
                        points=[point],
                        context=dict(cluster[-1].context),
                    )
                )
                i = j
                continue

        out.extend(cluster)
        i = j
    return out


def _is_scroll_action(action: str) -> bool:
    return bool(re.match(r"^Scroll(Down|Up)\s+x\d+\s+at\s+\(", action))


def _parse_scroll_action(
    action: str,
) -> tuple[Optional[str], int, Optional[dict[str, int]]]:
    m = re.match(r"^Scroll(Down|Up)\s+x(\d+)\s+at\s+\(([-\d]+),\s*([-\d]+)\)$", action)
    if not m:
        return None, 0, None
    direction = m.group(1).lower()
    count = int(m.group(2))
    point = {"x": int(m.group(3)), "y": int(m.group(4))}
    return direction, count, point


def _point_distance(a: CompactAction, b: CompactAction) -> float:
    if not a.points or not b.points:
        return 0.0
    ax, ay = a.points[-1]["x"], a.points[-1]["y"]
    bx, by = b.points[-1]["x"], b.points[-1]["y"]
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _is_typeable_key(key: str) -> bool:
    if key in TYPEABLE_SPECIAL:
        return True
    if len(key) == 1:
        return True
    return key in {"backspace"}


def _semantic_text_edit_cleanup(actions: list[CompactAction]) -> list[CompactAction]:
    out: list[CompactAction] = []
    for act in actions:
        if not act.action.startswith("Type:"):
            out.append(act)
            continue
        text = act.action.removeprefix("Type:").strip()
        if not text:
            continue
        bs_count = text.count("<BS>")
        if bs_count == 0:
            out.append(act)
            continue

        stripped = text.replace("<BS>", "").strip()
        if stripped:
            out.append(
                CompactAction(
                    t_start=act.t_start,
                    t_end=act.t_end,
                    action=f'Edit text: clear previous input and type "{stripped}"',
                    software=act.software,
                    points=act.points,
                    context=dict(act.context),
                )
            )
            continue

        out.append(
            CompactAction(
                t_start=act.t_start,
                t_end=act.t_end,
                action=f"Delete previous characters x{bs_count}",
                software=act.software,
                points=act.points,
                context=dict(act.context),
            )
        )
    return out


def _merge_hotkey_confirm_pairs(actions: list[CompactAction]) -> list[CompactAction]:
    out: list[CompactAction] = []
    i = 0
    nav_keys = {"UP", "DOWN", "LEFT", "RIGHT"}
    while i < len(actions):
        cur = actions[i]
        if not cur.action.startswith("Hotkey: "):
            out.append(cur)
            i += 1
            continue
        key = cur.action.removeprefix("Hotkey: ").strip()
        if key not in nav_keys or i + 1 >= len(actions):
            out.append(cur)
            i += 1
            continue
        nxt = actions[i + 1]
        if (
            nxt.action == "Press Enter"
            and nxt.software == cur.software
            and 0.0 <= (nxt.t_start - cur.t_end) <= 1.2
        ):
            out.append(
                CompactAction(
                    t_start=cur.t_start,
                    t_end=nxt.t_end,
                    action=f"Navigate with {key} and confirm",
                    software=cur.software,
                    points=[],
                    context=dict(nxt.context or cur.context),
                )
            )
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


def _normalize_key_to_char(key_raw: str) -> Optional[str]:
    key = key_raw.lower()
    if key in TYPEABLE_SPECIAL:
        return TYPEABLE_SPECIAL[key]
    if key == "backspace":
        return "<BS>"
    if len(key_raw) == 1:
        return key_raw
    return None


def _point(payload: dict[str, Any]) -> Optional[dict[str, int]]:
    if "x" in payload and "y" in payload:
        try:
            return {"x": int(payload["x"]), "y": int(payload["y"])}
        except Exception:
            return None
    return None


def _write_trace(actions: list[CompactAction], trace_path: Path) -> None:
    trajectory: list[dict[str, Any]] = []
    for idx, action in enumerate(actions, start=1):
        context = _clean_context(action.context)
        caption = _raw_caption(action.software, action.action, context=context)
        trajectory.append(
            {
                "step_idx": idx,
                "caption": caption,
                "context": context,
                "time_info": {
                    "start_time": round(action.t_start, 3),
                    "end_time": round(max(action.t_end, action.t_start + 0.01), 3),
                    "current_action_trigger_timestamp": round(action.t_end, 3),
                },
                "caption_prompt_input[Debug]": {
                    "action": [round(action.t_end, 3), action.action, action.points],
                    "context": context,
                },
            }
        )

    trace = {"trajectory": trajectory}
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _raw_caption(
    software: str, action_text: str, context: dict[str, Any]
) -> dict[str, Any]:
    """Minimal placeholder; online_captioner replaces this using marked screenshots."""
    return {
        "software_name": software or "UnknownApp",
        "window_title": context.get("window_title"),
        "bundle_id": context.get("bundle_id"),
        "url": context.get("url"),
        "ui_role": (
            (context.get("ui") or {}).get("role")
            if isinstance(context.get("ui"), dict)
            else None
        ),
        "observation_action_before": "The relevant UI is visible before the action.",
        "observation_action_after": "The UI updates after the action.",
        "think": "Use the marked screenshots to describe this action.",
        "action": action_text.strip(),
        "expectation": "The interface reflects the action result.",
    }


def _context_from_event(ev: dict[str, Any]) -> dict[str, Any]:
    ctx = ev.get("context")
    if not isinstance(ctx, dict):
        ctx = {}
    ui = ctx.get("ui")
    if not isinstance(ui, dict):
        ui = ev.get("ui") if isinstance(ev.get("ui"), dict) else {}
    return {
        "app_name": ev.get("window") or ctx.get("app_name"),
        "window_title": ev.get("window_title") or ctx.get("window_title"),
        "bundle_id": ev.get("bundle_id") or ctx.get("bundle_id"),
        "pid": ev.get("pid") or ctx.get("pid"),
        "url": ev.get("url") or ctx.get("url"),
        "ui": ui or None,
    }


def _clean_context(context: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("app_name", "window_title", "bundle_id", "pid", "url", "ui"):
        val = context.get(key)
        if val is None:
            continue
        if isinstance(val, str) and not val.strip():
            continue
        out[key] = val
    return out
