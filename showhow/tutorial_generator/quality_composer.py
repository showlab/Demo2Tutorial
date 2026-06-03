from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)

# =====================================================================
# quality_composer.py
# Pipeline: parse → plan/critic → [this file] annotate + compose
#
# Frame extraction : cv2  (best-frame selection, adaptive zoom/crop)
# Annotation       : PIL  (badge, instruction box, click marker,
#                          drag path, SAM2 highlight — frame_annotator style)
# Output           : annotated PNG per step + index.md + dataset.jsonl
# =====================================================================


# ─────────────────────────── SAM2 ────────────────────────────────────


class SAM2Wrapper:
    """Point-prompt → bbox.  Config comes from ComposerConfig.sam2."""

    def __init__(
        self, checkpoint: str = "", config_yaml: str = "", device: str = "cuda"
    ):
        self.predictor = None
        self._torch = None
        if not checkpoint or not config_yaml:
            return
        try:
            import torch
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            self.predictor = SAM2ImagePredictor(build_sam2(config_yaml, checkpoint))
            self._torch = torch
            logger.info("[SAM2] Ready.")
        except Exception as e:
            logger.warning("[SAM2] Init failed: %s", e)

    def available(self) -> bool:
        return self.predictor is not None

    def predict_bbox_from_point(
        self, img_rgb: np.ndarray, pt: tuple[int, int]
    ) -> Optional[tuple[int, int, int, int]]:
        if not self.available():
            return None
        torch = self._torch
        try:
            ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else torch.autocast("cpu", dtype=torch.bfloat16)
            )
            with torch.inference_mode(), ctx:
                self.predictor.set_image(img_rgb)
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([[pt]], dtype=np.float32),
                    point_labels=np.array([[1]], dtype=np.int32),
                )
            if masks is None or len(masks) == 0:
                return None
            k = int(np.argmax(scores))
            m = masks[k] > 0
            ys, xs = np.where(m)
            if ys.size == 0:
                return None
            return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        except Exception as e:
            logger.debug("[SAM2] predict failed: %s", e)
            return None


# ─────────────────────────── Action parsing ──────────────────────────


def _parse_raw_action(
    raw_action: Any,
) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (action_types, click_points, drag_points)."""
    types: list[str] = []
    clicks: list[tuple[int, int]] = []
    drag: list[tuple[int, int]] = []

    if raw_action is None:
        return types, clicks, drag

    def _pts(raw) -> list[tuple[int, int]]:
        out = []
        if not isinstance(raw, list):
            return out
        for p in raw:
            if isinstance(p, dict) and "x" in p and "y" in p:
                out.append((int(p["x"]), int(p["y"])))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((int(p[0]), int(p[1])))
        return out

    def _walk(item: Any) -> None:
        if isinstance(item, list):
            for sub_item in item:
                _walk(sub_item)
            return
        if not isinstance(item, dict):
            return

        # Planner outputs may wrap the actual geometry under a nested raw_action field.
        nested = item.get("raw_action")
        if isinstance(nested, (dict, list)):
            _walk(nested)

        t = str(item.get("type", "")).strip()
        if t:
            types.append(t)
        pts = _pts(item.get("points") or [])
        if pts:
            if "drag" in t.lower() or len(pts) > 1:
                drag.extend(pts)
            else:
                clicks.extend(pts)

    _walk(raw_action)
    return types, clicks, drag


def _extract_focus_points(
    raw_action: Any,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """Return (all_points, primary_points, primary_click_count)."""
    _, clicks, drag = _parse_raw_action(raw_action)
    all_points = clicks + drag

    if isinstance(raw_action, list):
        for item in reversed(raw_action):
            if not isinstance(item, dict):
                continue
            nested = item.get("raw_action")
            if not isinstance(nested, (dict, list)):
                continue
            _, primary_clicks, primary_drag = _parse_raw_action(nested)
            primary_points = primary_clicks + primary_drag
            if primary_points:
                return all_points, primary_points, len(primary_clicks)

    return all_points, list(all_points), len(clicks)


def classify_label(types: list[str]) -> str:
    tl = [t.lower() for t in types]
    if any("drag" in t for t in tl):
        return "Drag"
    if any("rclick" in t or "lclick" in t or "click" in t for t in tl):
        return "Click"
    if any(t.startswith("hotkey") for t in tl):
        for t in types:
            if t.lower().startswith("hotkey") and ":" in t:
                return "Hotkey: " + t.split(":", 1)[1].strip()
        return "Hotkey"
    if any(t.startswith("type") for t in tl):
        for t in types:
            if t.lower().startswith("type") and ":" in t:
                return "Type: " + t.split(":", 1)[1].strip()
        return "Type"
    if any("key press" in t for t in tl):
        for t in types:
            if "key press" in t.lower() and ":" in t:
                return "Key: " + t.split(":", 1)[1].strip()
        return "Key"
    return "Step"


def _label_to_view_kind(label: str) -> str:
    """Map classify_label output to the kind string used by _apply_adaptive_view."""
    if label == "Drag":
        return "drag"
    if label == "Click":
        return "click"
    if label.startswith("Hotkey") or label.startswith("Key"):
        return "hotkey"
    if label.startswith("Type"):
        return "type"
    if label == "Step":
        return "generic"
    return "generic"


# ─────────────────────────── PIL annotation ──────────────────────────

_MARGIN = 16
_BADGE_FONT_SIZE = 22
_INSTR_FONT_SIZE = 20
_CLICK_RADIUS = 22
_CLICK_COLOR = (255, 230, 0, 255)
_DRAG_COLOR = (120, 200, 255, 220)
_HL_FILL = (255, 200, 50, 35)
_HL_STROKE = (255, 200, 50, 210)
_HL_STROKE_W = 3
_DEBUG_COMPOSER = False


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, font: Any, max_width: int) -> str:
    words = str(text).split(" ")
    lines: list[str] = []
    cur = ""
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    for w in words:
        test = (cur + " " + w).strip()
        if dummy.textlength(test, font=font) > max_width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def draw_badge(im: Image.Image, text: str) -> None:
    font = _load_font(_BADGE_FONT_SIZE)
    pad = 12
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw, th = dummy.textbbox((0, 0), text, font=font)[2:]
    w, h = tw + pad * 2, th + pad * 2
    badge = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bd = ImageDraw.Draw(badge, "RGBA")
    bd.rounded_rectangle(
        (0, 0, w, h),
        radius=14,
        fill=(20, 20, 20, 210),
        outline=(255, 255, 255, 60),
        width=2,
    )
    bd.text((pad, pad), text, font=font, fill=(255, 255, 255, 230))
    im.alpha_composite(badge, (_MARGIN, _MARGIN))


def draw_instruction(im: Image.Image, text: str) -> None:
    W, H = im.size
    font = _load_font(_INSTR_FONT_SIZE)
    wrapped = _wrap_text(text, font, int(W * 0.82))
    dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bb = dummy.multiline_textbbox((0, 0), wrapped, font=font, spacing=6)
    iw, ih = bb[2] - bb[0], bb[3] - bb[1]
    pad = 16
    bw, bh = iw + pad * 2, ih + pad * 2
    ov = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov, "RGBA")
    od.rounded_rectangle(
        (0, 0, bw, bh),
        radius=16,
        fill=(0, 0, 0, 155),
        outline=(255, 255, 255, 40),
        width=2,
    )
    od.multiline_text(
        (pad, pad), wrapped, font=font, fill=(255, 255, 255, 235), spacing=6
    )
    im.alpha_composite(ov, (int((W - bw) // 2), H - bh - _MARGIN))


def click_marker(
    im: Image.Image,
    center: tuple[int, int],
    radius: int = _CLICK_RADIUS,
    marker_text: Optional[str] = None,
) -> None:
    x, y = center
    W, H = im.size
    r, g, b, _ = _CLICK_COLOR

    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(halo, "RGBA").ellipse(
        (x - radius - 18, y - radius - 18, x + radius + 18, y + radius + 18),
        fill=(r, g, b, 92),
    )
    halo = halo.filter(ImageFilter.GaussianBlur(10))
    im.alpha_composite(halo)

    ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ring_draw = ImageDraw.Draw(ring, "RGBA")
    ring_draw.ellipse(
        (x - radius - 6, y - radius - 6, x + radius + 6, y + radius + 6),
        outline=(255, 255, 255, 170),
        width=3,
    )
    ring_draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=(r, g, b, 235),
        width=6,
    )
    im.alpha_composite(ring)

    dot = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dot_draw = ImageDraw.Draw(dot, "RGBA")
    dot_draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(r, g, b, 240))
    dot_draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255, 255))
    im.alpha_composite(dot)

    if marker_text:
        font = _load_font(16)
        pad_x, pad_y = 7, 4
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bb = dummy.textbbox((0, 0), marker_text, font=font)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        bw, bh = tw + pad_x * 2, th + pad_y * 2
        bx = min(max(8, x + radius - 2), max(8, W - bw - 8))
        by = min(max(8, y - radius - bh - 6), max(8, H - bh - 8))
        bubble = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        bubble_draw = ImageDraw.Draw(bubble, "RGBA")
        bubble_draw.rounded_rectangle(
            (bx, by, bx + bw, by + bh),
            radius=10,
            fill=(28, 28, 28, 230),
            outline=(255, 255, 255, 70),
            width=2,
        )
        bubble_draw.text(
            (bx + pad_x, by + pad_y - 1),
            marker_text,
            font=font,
            fill=(255, 255, 255, 245),
        )
        im.alpha_composite(bubble)


def draw_drag(im: Image.Image, pts: list[tuple[int, int]]) -> None:
    if len(pts) < 2:
        return
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay, "RGBA")
    d.line(pts, fill=(255, 255, 255, 110), width=9)
    d.line(pts, fill=_DRAG_COLOR, width=6)
    sx, sy = pts[0]
    ex, ey = pts[-1]
    sr, er = 10, 12
    d.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=(255, 255, 255, 235))
    d.ellipse((sx - sr + 3, sy - sr + 3, sx + sr - 3, sy + sr - 3), fill=_DRAG_COLOR)
    d.ellipse((ex - er, ey - er, ex + er, ey + er), fill=_DRAG_COLOR)

    if len(pts) >= 2:
        px, py = pts[-2]
        dx, dy = ex - px, ey - py
        norm = max(1.0, float((dx * dx + dy * dy) ** 0.5))
        ux, uy = dx / norm, dy / norm
        perp_x, perp_y = -uy, ux
        arrow_len = 20
        wing = 10
        tip = (ex, ey)
        left = (
            int(round(ex - ux * arrow_len + perp_x * wing)),
            int(round(ey - uy * arrow_len + perp_y * wing)),
        )
        right = (
            int(round(ex - ux * arrow_len - perp_x * wing)),
            int(round(ey - uy * arrow_len - perp_y * wing)),
        )
        d.polygon([tip, left, right], fill=_DRAG_COLOR)

    im.alpha_composite(overlay)


def draw_highlight_box(im: Image.Image, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    ImageDraw.Draw(im, "RGBA").rectangle(
        (x1, y1, x2, y2),
        outline=_HL_STROKE,
        width=_HL_STROKE_W,
        fill=_HL_FILL,
    )


def _draw_debug_overlay(
    im: Image.Image,
    all_points: list[tuple[int, int]],
    primary_points: list[tuple[int, int]],
    focus_box: Optional[tuple[int, int, int, int]],
) -> None:
    if not _DEBUG_COMPOSER:
        return
    debug = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(debug, "RGBA")
    if focus_box:
        d.rectangle(focus_box, outline=(255, 80, 80, 230), width=3)
    for x, y in all_points:
        d.line((x - 8, y, x + 8, y), fill=(255, 255, 255, 220), width=2)
        d.line((x, y - 8, x, y + 8), fill=(255, 255, 255, 220), width=2)
    for x, y in primary_points:
        d.ellipse(
            (x - 10, y - 10, x + 10, y + 10),
            outline=(255, 80, 80, 240),
            width=3,
        )
    im.alpha_composite(debug)


# ─────────────────────────── Frame extraction (cv2) ──────────────────


def _extract_resolution_from_video(video_path: Path) -> Optional[tuple[int, int]]:
    decrypted_log = video_path.parent / video_path.stem / "operations_decrypted.txt"
    if not decrypted_log.exists():
        return None
    try:
        first_line = decrypted_log.read_text(encoding="utf-8").splitlines()[0]
        data = json.loads(first_line)
        config_data = json.loads(data.get("message", "{}"))
        first_key = next(iter(config_data), None)
        if (
            first_key
            and "width" in config_data[first_key]
            and "height" in config_data[first_key]
        ):
            return (
                int(config_data[first_key]["width"]),
                int(config_data[first_key]["height"]),
            )
    except Exception:
        pass
    return None


def _scale_points(
    points: list[tuple[int, int]], raw_coord_space: tuple[int, int], frame: Any
) -> list[tuple[int, int]]:
    raw_w, raw_h = raw_coord_space
    if raw_w <= 0 or raw_h <= 0:
        return points
    h, w = frame.shape[:2]
    sx, sy = w / raw_w, h / raw_h
    return [(int(round(x * sx)), int(round(y * sy))) for x, y in points]


def _focus_box(
    frame: Any, points: list[tuple[int, int]], cv2: Any
) -> Optional[tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    if not points:
        return None
    xs = [min(max(0, p[0]), w - 1) for p in points]
    ys = [min(max(0, p[1]), h - 1) for p in points]
    pad = int(max(60, 0.06 * min(w, h)))
    return (
        max(0, min(xs) - pad),
        max(0, min(ys) - pad),
        min(w - 1, max(xs) + pad),
        min(h - 1, max(ys) + pad),
    )


def _apply_adaptive_view(
    frame: Any,
    points: list[tuple[int, int]],
    focus_box: Optional[tuple[int, int, int, int]],
    cv2: Any,
    view_kind: str,
) -> tuple[Any, list[tuple[int, int]]]:
    h, w = frame.shape[:2]
    if not focus_box:
        return frame, points
    x1, y1, x2, y2 = focus_box
    fw, fh = max(1, x2 - x1), max(1, y2 - y1)
    area_ratio = (fw * fh) / float(max(1, w * h))

    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        frame[y1:y2, x1:x2] = cv2.convertScaleAbs(roi, alpha=1.18, beta=10)

    if area_ratio < 0.22 and view_kind in {"click", "type", "hotkey", "generic"}:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cw, ch = int(w * 0.68), int(h * 0.68)
        sx1 = max(0, min(w - cw, cx - cw // 2))
        sy1 = max(0, min(h - ch, cy - ch // 2))
        crop = frame[sy1 : sy1 + ch, sx1 : sx1 + cw]
        if crop.size > 0:
            frame = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
            sx, sy = w / float(cw), h / float(ch)
            points = [
                (
                    min(max(0, int(round((px - sx1) * sx))), w - 1),
                    min(max(0, int(round((py - sy1) * sy))), h - 1),
                )
                for px, py in points
            ]
    return frame, points


def _extract_best_frame(
    cap: Any, cv2: Any, t_hint: Any, fps: float, focus_points: list[tuple[int, int]]
) -> Any:
    if cap is None or cv2 is None or not cap.isOpened():
        return None
    try:
        t = max(0.0, float(t_hint) if t_hint is not None else 0.0)
    except Exception:
        t = 0.0
    if fps <= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        return frame if ok else None

    cache: dict[int, Any] = {}
    candidates: list[tuple[float, Any]] = []
    for off in [-0.6, -0.4, -0.25, -0.12, 0.0, 0.12, 0.25, 0.4, 0.6]:
        idx = int(max(0.0, t + off) * fps)
        frame = _read_frame_at(cap, cv2, idx, cache)
        if frame is None:
            continue
        score = _frame_quality_score(
            frame=frame,
            cv2=cv2,
            t_offset=abs(off),
            prev_frame=_read_frame_at(cap, cv2, max(0, idx - 2), cache),
            next_frame=_read_frame_at(cap, cv2, idx + 2, cache),
            focus_points=focus_points,
        )
        candidates.append((score, frame))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    return frame if ok else None


def _read_frame_at(cap: Any, cv2: Any, idx: int, cache: dict[int, Any]) -> Any:
    if idx in cache:
        return cache[idx]
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cache[idx] = frame if ok else None
    return cache[idx]


def _frame_quality_score(
    frame: Any,
    cv2: Any,
    t_offset: float,
    prev_frame: Any,
    next_frame: Any,
    focus_points: list[tuple[int, int]],
) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharp_n = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 300.0)
    proximity = max(0.0, 1.0 - t_offset / 0.8)
    motion_penalty = 0.0
    if prev_frame is not None and next_frame is not None:
        p = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        n = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        d1 = float(cv2.absdiff(gray, p).mean())
        d2 = float(cv2.absdiff(gray, n).mean())
        motion_penalty = min(1.0, ((d1 + d2) / 2.0) / 28.0)
    return (
        0.45 * sharp_n
        + 0.20 * proximity
        + 0.20 * (1.0 - motion_penalty)
        + 0.15 * _focus_signal(frame, focus_points, cv2)
    )


def _focus_signal(frame: Any, focus_points: list[tuple[int, int]], cv2: Any) -> float:
    if not focus_points:
        return 0.5
    h, w = frame.shape[:2]
    vals = []
    for x, y in focus_points[:3]:
        x, y = min(max(0, x), w - 1), min(max(0, y), h - 1)
        patch = frame[max(0, y - 35) : min(h, y + 35), max(0, x - 35) : min(w, x + 35)]
        if patch.size == 0:
            continue
        vals.append(
            float(
                cv2.Laplacian(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            )
        )
    return min(1.0, (sum(vals) / len(vals)) / 220.0) if vals else 0.5


def _fallback_bbox(
    pt: tuple[int, int], w: int, h: int, r: int = 26
) -> tuple[int, int, int, int]:
    x, y = int(pt[0]), int(pt[1])
    return (max(0, x - r), max(0, y - r), min(w - 1, x + r), min(h - 1, y + r))


_N_CANDIDATES = 5


def _sample_candidate_frames(
    cap: Any, cv2: Any, fps: float, t_start: float, t_end: float, n: int = _N_CANDIDATES
) -> list[tuple[float, Any]]:
    """Sample n frames uniformly from [t_start, t_end], expanding if the interval is short."""
    if cap is None or cv2 is None or not cap.isOpened() or fps <= 0:
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count > 0:
        max_t = max(0.0, (frame_count - 1) / fps)
        t_start = min(max(0.0, t_start), max_t)
        t_end = min(max(t_start, t_end), max_t)
    duration = t_end - t_start
    if duration < 0.5:
        mid = (t_start + t_end) / 2.0
        t_start = max(0.0, mid - 0.25)
        t_end = mid + 0.25
        duration = t_end - t_start
    if n == 1:
        times = [(t_start + t_end) / 2.0]
    else:
        times = [t_start + duration * i / (n - 1) for i in range(n)]
    frames: list[tuple[float, Any]] = []
    for t in times:
        idx = int(max(0.0, t) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((t, frame))
    return frames


# ─────────────────────────── Main compose ────────────────────────────


def compose_quality_tutorial(
    video_path: Path,
    draft_json_path: Path,
    output_dir: Path,
    raw_coord_space: Optional[tuple[int, int]] = None,
    sam2_config: Any = None,
) -> tuple[Path, Optional[Path], Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(draft_json_path.read_text(encoding="utf-8"))
    logger.info("Composer: %d chapters → %s", len(data.get("chapters", [])), output_dir)

    images_dir = output_dir / "step_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "index.md"
    dataset_path = output_dir / "dataset.jsonl"

    # cv2 for frame extraction
    cv2 = None
    try:
        import cv2 as _cv2  # type: ignore

        cv2 = _cv2
    except Exception:
        pass

    # SAM2 for highlight bbox
    if sam2_config is not None and getattr(sam2_config, "enabled", False):
        sam2 = SAM2Wrapper(
            checkpoint=sam2_config.checkpoint,
            config_yaml=sam2_config.config_yaml,
            device=sam2_config.device,
        )
    else:
        sam2 = SAM2Wrapper()

    if raw_coord_space is None:
        raw_coord_space = _extract_resolution_from_video(video_path)

    cap = None
    fps = 0.0
    if cv2 is not None and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap is not None and cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    md_lines = [f"# {data.get('tutorial_goal', 'ShowHow Tutorial')}", ""]
    global_step = 0

    with dataset_path.open("w", encoding="utf-8") as jf:
        for ch_idx, chapter in enumerate(data.get("chapters", []), start=1):
            title = chapter.get("chapter_title", f"Chapter {ch_idx}")
            md_lines += [f"## {ch_idx}. {title}", ""]

            for step_idx, action in enumerate(chapter.get("actions", []), start=1):
                global_step += 1
                instruction = str(action.get("instruction", f"Step {step_idx}"))
                rationale = str(action.get("rationale", "")).strip()
                expected = str(action.get("expected_result", "")).strip()

                # ── parse action ──────────────────────────────────────
                raw_action = action.get("raw_action")
                types, clicks_raw, drag_raw = _parse_raw_action(raw_action)
                label = classify_label(types)
                view_kind = _label_to_view_kind(label)

                # all points are annotated, but the last nested action drives focus.
                all_raw, primary_raw, primary_click_count = _extract_focus_points(
                    raw_action
                )
                n_clicks = len(clicks_raw)

                # ── sample candidate frames (cv2) ─────────────────────
                t_start_v = float(action.get("t_start") or 0.0)
                t_end_v = float(action.get("t_end") or t_start_v)
                candidate_frames = _sample_candidate_frames(
                    cap, cv2, fps, t_start_v, t_end_v
                )

                image_rel: Optional[str] = None
                candidate_images: list[str] = []
                clicks_s: list[tuple[int, int]] = []
                drag_s: list[tuple[int, int]] = []

                if candidate_frames and cv2 is not None:
                    # compute scaled points once from first frame (all same resolution)
                    _, ref_frame = candidate_frames[0]
                    all_pts = (
                        _scale_points(all_raw, raw_coord_space, ref_frame)
                        if raw_coord_space and all_raw
                        else list(all_raw)
                    )
                    primary_pts = (
                        _scale_points(primary_raw, raw_coord_space, ref_frame)
                        if raw_coord_space and primary_raw
                        else list(primary_raw)
                    )
                    focus_pts = primary_pts if primary_pts else all_pts
                    focus_box = _focus_box(ref_frame, focus_pts, cv2)

                    for ci, (_, frame_c) in enumerate(candidate_frames):
                        frame_a = frame_c.copy()
                        frame_a, all_pts_r = _apply_adaptive_view(
                            frame_a, all_pts, focus_box, cv2, view_kind
                        )
                        clicks_s = all_pts_r[:n_clicks]
                        drag_s = all_pts_r[n_clicks:]
                        h_f, w_f = frame_a.shape[:2]

                        # ── convert to PIL RGBA for annotation ───────
                        im = Image.fromarray(
                            cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
                        ).convert("RGBA")
                        frame_rgb = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
                        _, primary_pts_r = _apply_adaptive_view(
                            frame_c.copy(), focus_pts, focus_box, cv2, view_kind
                        )
                        _draw_debug_overlay(im, all_pts_r, primary_pts_r, focus_box)

                        # SAM2 highlight boxes (click points only)
                        for pt in clicks_s[:3]:
                            bbox = sam2.predict_bbox_from_point(frame_rgb, pt)
                            draw_highlight_box(
                                im, bbox if bbox else _fallback_bbox(pt, w_f, h_f)
                            )

                        # drag path
                        if len(drag_s) >= 2:
                            draw_drag(im, drag_s)

                        # click markers
                        primary_click_pt = (
                            primary_pts_r[primary_click_count - 1]
                            if primary_click_count > 0
                            and len(primary_pts_r) >= primary_click_count
                            else None
                        )
                        for idx, pt in enumerate(clicks_s[:3], start=1):
                            click_marker(
                                im,
                                pt,
                                marker_text=str(idx) if pt == primary_click_pt else None,
                            )

                        # badge (top-left) + instruction (bottom centre)
                        draw_badge(im, label)
                        draw_instruction(im, instruction)

                        c_name = f"step_{global_step:04d}_c{ci}.png"
                        im.convert("RGB").save(str(images_dir / c_name))
                        candidate_images.append(f"step_images/{c_name}")

                    # middle candidate is the default displayed image
                    mid = len(candidate_images) // 2
                    image_rel = candidate_images[mid] if candidate_images else None

                    # recompute final points for dataset (from middle candidate)
                    if candidate_frames:
                        _, ref_frame = candidate_frames[0]
                        all_pts_f = (
                            _scale_points(all_raw, raw_coord_space, ref_frame)
                            if raw_coord_space and all_raw
                            else list(all_raw)
                        )
                        _, all_pts_f = _apply_adaptive_view(
                            ref_frame.copy(), all_pts_f, focus_box, cv2, view_kind
                        )
                        clicks_s = all_pts_f[:n_clicks]
                        drag_s = all_pts_f[n_clicks:]
                    else:
                        clicks_s, drag_s = [], []

                # ── markdown ──────────────────────────────────────────
                md_lines.append(f"### Step {ch_idx}.{step_idx}: {instruction}")
                if rationale:
                    md_lines.append(f"- Why: {rationale}")
                if expected:
                    md_lines.append(f"- Expected: {expected}")
                if image_rel:
                    md_lines += ["", f"![{instruction}]({image_rel})"]
                md_lines.append("")

                # ── dataset record ────────────────────────────────────
                has_frames = bool(candidate_images)
                record = {
                    "chapter_index": ch_idx,
                    "step_index": step_idx,
                    "chapter_title": title,
                    "instruction": instruction,
                    "rationale": rationale,
                    "expected_result": expected,
                    "image_path": image_rel,
                    "candidate_images": candidate_images,
                    "t_start": action.get("t_start"),
                    "t_end": action.get("t_end"),
                    "original_id": action.get("original_id", ""),
                    "context": action.get("context", {}),
                    "highlight": action.get("highlight", {}),
                    "highlight_metadata": {
                        "label": label,
                        "clicks": [{"x": x, "y": y} for x, y in clicks_s]
                        if has_frames
                        else [],
                        "drag": [{"x": x, "y": y} for x, y in drag_s]
                        if has_frames
                        else [],
                    },
                }
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass

    markdown_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Composer: done — %d steps → %s", global_step, output_dir)
    return output_dir, markdown_path, dataset_path
