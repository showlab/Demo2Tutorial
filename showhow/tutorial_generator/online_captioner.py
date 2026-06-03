from __future__ import annotations

import base64
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from .openai_compat import chat_completion_create

logger = logging.getLogger(__name__)


_SYSTEM = """You are a desktop tutorial step captioner.
Given screenshots and a raw action description, write ONE concise sentence describing what
the user did and where — like a tutorial instruction.

Prioritize the visual screenshots over app/window metadata. The before/after screenshots
may contain the same red/yellow marker showing the action location; describe the UI element
at that marker location. Do not infer "taskbar", "system tray", or app names from metadata
unless the marked screenshots support it.

Examples:
- "Click the 'File' menu to open it."
- "Type 'Hello World' into the search box."
- "Press Ctrl+S to save the document."
- "Scroll down in the layers panel."
- "Drag the slider to adjust the volume."

Return STRICT JSON only:
{"caption": "<one sentence instruction>"}"""


def enhance_trace_captions_online(trace_json_path: Path, model: str = "gpt-4o") -> Path:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for online captioning")

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key)

    data = json.loads(trace_json_path.read_text(encoding="utf-8"))
    trajectory: list[dict[str, Any]] = data.get("trajectory", [])
    logger.info("Captioner: %d steps, model=%s", len(trajectory), model)

    cap, cv2, fps = _open_video(trace_json_path.parent)
    if cap is None:
        raise RuntimeError(
            "Captioner requires a session video for screenshot-based captions"
        )
    logger.info("Captioner: vision mode (per-step screenshots)")

    success = 0
    for step in trajectory:
        caption_text = _caption_step(client, model, step, cap, cv2, fps)
        if caption_text:
            step.setdefault("caption", {})["action"] = caption_text
            success += 1

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    trace_json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Captioner: done — %d/%d steps captioned", success, len(trajectory))
    if trajectory and success == 0:
        raise RuntimeError("Captioner failed to caption every step")
    return trace_json_path


def _caption_step(
    client: Any,
    model: str,
    step: dict[str, Any],
    cap: Any,
    cv2: Any,
    fps: float,
) -> Optional[str]:
    caption = step.get("caption", {}) or {}
    action_str = caption.get("action", "")
    software = caption.get("software_name", "")
    context = step.get("context", {}) or {}
    window = context.get("window_title") or software or ""

    action_desc = f"[{window}] {action_str}" if window else action_str

    time_info = step.get("time_info", {}) or {}
    t_start = float(time_info.get("start_time", 0.0))
    t_end = float(time_info.get("end_time", t_start))
    raw_points = _debug_points(step)
    b64_before = _grab_frame_b64(
        cap, cv2, fps, max(0.0, t_start - 0.1), points=raw_points, trace_context=step
    )
    b64_after = _grab_frame_b64(
        cap, cv2, fps, t_end + 0.3, points=raw_points, trace_context=step
    )
    marker_note = (
        " Both screenshots include a red/yellow marker at the action location."
        if raw_points
        else ""
    )
    content: list[Any] = [
        {"type": "text", "text": f"Action: {action_desc}\n{marker_note}\n\nBefore:"}
    ]
    if b64_before:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_before}",
                    "detail": "high",
                },
            }
        )
    content.append({"type": "text", "text": "After:"})
    if b64_after:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_after}",
                    "detail": "high",
                },
            }
        )

    try:
        resp = chat_completion_create(
            client,
            model=model,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": content},
            ],
        )
        txt = (resp.choices[0].message.content or "").strip()
        parsed = _parse(txt)
        return parsed.get("caption") or None
    except Exception as e:
        logger.warning("Caption failed for step %s: %s", step.get("step_idx"), e)
        return None


def _open_video(session_dir: Path):
    try:
        import cv2 as _cv2  # type: ignore
    except ImportError:
        return None, None, 0.0

    for ext in ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm"):
        hits = sorted(session_dir.glob(ext))
        if hits:
            cap = _cv2.VideoCapture(str(hits[0]))
            if cap.isOpened():
                fps = float(cap.get(_cv2.CAP_PROP_FPS) or 0.0)
                return cap, _cv2, fps
    return None, None, 0.0


def _grab_frame_b64(
    cap: Any,
    cv2: Any,
    fps: float,
    t: float,
    *,
    points: Optional[list[dict[str, int]]] = None,
    trace_context: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    if fps <= 0:
        return None
    idx = max(0, int(t * fps))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        idx = min(idx, frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    try:
        from PIL import Image  # type: ignore

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if points:
            _draw_action_markers(img, points, trace_context or {})
        w, h = img.size
        if w > 1024:
            img = img.resize((1024, int(h * 1024 / w)))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _debug_points(step: dict[str, Any]) -> list[dict[str, int]]:
    debug = step.get("caption_prompt_input[Debug]", {}) or {}
    action = debug.get("action")
    if not isinstance(action, list) or len(action) < 3:
        return []
    raw = action[2]
    if not isinstance(raw, list):
        return []
    points: list[dict[str, int]] = []
    for p in raw:
        if isinstance(p, dict) and "x" in p and "y" in p:
            try:
                points.append({"x": int(p["x"]), "y": int(p["y"])})
            except Exception:
                continue
    return points


def _draw_action_markers(
    img: Any, points: list[dict[str, int]], step: dict[str, Any]
) -> None:
    from PIL import ImageDraw  # type: ignore

    draw = ImageDraw.Draw(img)
    w, h = img.size
    scaled = [_scale_point_to_frame(p, w, h, step) for p in points]
    scaled = [p for p in scaled if p is not None]
    if not scaled:
        return

    if len(scaled) >= 2:
        draw.line(scaled, fill=(255, 210, 0), width=8)
        draw.line(scaled, fill=(220, 0, 0), width=3)

    for x, y in scaled[:4]:
        r = max(18, min(w, h) // 55)
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 210, 0), width=8)
        draw.ellipse(
            (x - r + 5, y - r + 5, x + r - 5, y + r - 5), outline=(220, 0, 0), width=4
        )
        draw.line((x - r - 8, y, x + r + 8, y), fill=(220, 0, 0), width=3)
        draw.line((x, y - r - 8, x, y + r + 8), fill=(220, 0, 0), width=3)


def _scale_point_to_frame(
    point: dict[str, int],
    frame_w: int,
    frame_h: int,
    step: dict[str, Any],
) -> Optional[tuple[int, int]]:
    try:
        x = int(point["x"])
        y = int(point["y"])
    except Exception:
        return None

    raw_w, raw_h = _coord_space_from_step(step)
    if raw_w and raw_h:
        x = int(round(x * frame_w / raw_w))
        y = int(round(y * frame_h / raw_h))
    elif x <= frame_w // 2 + 80 and y <= frame_h // 2 + 80:
        # macOS Retina fallback: pynput coordinates are often logical points,
        # while AVFoundation frames are physical pixels.
        x *= 2
        y *= 2

    return (
        min(max(0, x), frame_w - 1),
        min(max(0, y), frame_h - 1),
    )


def _coord_space_from_step(step: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    screen = (step.get("context") or {}).get("screen") or {}
    coord = screen.get("coordinate_space") if isinstance(screen, dict) else None
    if isinstance(coord, dict):
        try:
            w = int(coord.get("width") or 0)
            h = int(coord.get("height") or 0)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    return None, None


def _parse(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except Exception:
                pass
        return {}
