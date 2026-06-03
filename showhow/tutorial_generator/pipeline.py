from __future__ import annotations

import json
import logging
import platform
import subprocess
from pathlib import Path
from typing import Optional

from .backend_demo2tut import Demo2TutBackend, is_critic_approved
from .config import GeneratorConfig
from .types import GenerateResult, PlanResult

logger = logging.getLogger(__name__)


def _probe_video_size(video_path: Path) -> Optional[tuple[int, int]]:
    """Return (width, height) of the first video stream via ffprobe."""
    for bin_name in ("ffprobe", "ffmpeg"):
        try:
            if bin_name == "ffprobe":
                args = [
                    bin_name,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(video_path),
                ]
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=10, check=False
                )
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                if streams:
                    w, h = streams[0].get("width"), streams[0].get("height")
                    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
                        return (w, h)
            else:
                # ffmpeg -i prints info to stderr, no output file needed
                args = [bin_name, "-v", "error", "-i", str(video_path)]
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=10, check=False
                )
                import re

                m = re.search(r"(\d{3,5})x(\d{3,5})", result.stderr)
                if m:
                    w, h = int(m.group(1)), int(m.group(2))
                    if w > 0 and h > 0:
                        return (w, h)
        except Exception:
            continue
    return None


def _read_recording_coord_space(data_path: Path) -> Optional[tuple[int, int]]:
    metadata_path = data_path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        coord = (metadata.get("screen") or {}).get("coordinate_space") or {}
        width = int(coord.get("width") or 0)
        height = int(coord.get("height") or 0)
        if width > 0 and height > 0:
            return (width, height)
    except Exception:
        return None
    return None


def _infer_recording_coord_space(
    data_path: Path, video_size: Optional[tuple[int, int]]
) -> Optional[tuple[int, int]]:
    if not video_size:
        return None
    metadata_path = data_path / "metadata.json"
    try:
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
    except Exception:
        metadata = {}
    if "macos" not in str(metadata.get("platform") or platform.platform()).lower():
        return None

    events_path = data_path / "events.jsonl"
    if not events_path.exists():
        return None
    max_x = 0
    max_y = 0
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    payload = json.loads(line).get("payload") or {}
                    max_x = max(max_x, int(payload.get("x") or 0))
                    max_y = max(max_y, int(payload.get("y") or 0))
                except Exception:
                    continue
    except Exception:
        return None

    video_w, video_h = video_size
    half_w = max(1, int(round(video_w / 2)))
    half_h = max(1, int(round(video_h / 2)))
    if max_x <= half_w + 80 and max_y <= half_h + 80:
        return (half_w, half_h)
    return None


class TutorialGenerator:
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        backend: Optional[Demo2TutBackend] = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.backend = backend or Demo2TutBackend()
        if hasattr(self.backend, "configure_modes"):
            self.backend.configure_modes(
                caption_model=self.config.caption_model,
            )

    def parse(self, video_path: str | Path, on_stage=None):
        return self.backend.parse_actions(Path(video_path).resolve(), on_stage=on_stage)

    def plan(self, trace_json_path: str | Path, on_stage=None) -> PlanResult:
        trace_path = Path(trace_json_path).resolve()
        planner_cfg = self.config.planner
        critic_cfg = self.config.critic

        draft_path: Optional[Path] = None
        approved = not critic_cfg.enabled
        critic_feedback: Optional[str] = None
        iterations = 0
        max_iter = critic_cfg.max_iterations

        for iter_idx in range(1, max_iter + 1):
            iterations = iter_idx
            if on_stage:
                on_stage(f"plan (iter {iter_idx})")
            logger.info("Plan iter %d/%d — calling planner LLM", iter_idx, max_iter)
            draft_path = self.backend.plan_once(
                trace_json_path=trace_path,
                planner_cfg=planner_cfg,
                critic_feedback=critic_feedback,
                iteration_id=iter_idx,
            )
            if not critic_cfg.enabled:
                approved = True
                break

            logger.info(
                "Plan iter %d/%d — evaluating draft with critic", iter_idx, max_iter
            )
            critic_output = self.backend.evaluate_draft(
                draft_json_path=draft_path,
                critic_cfg=critic_cfg,
                iteration_id=iter_idx,
            )
            approved = is_critic_approved(critic_output)
            critic_feedback = critic_output
            logger.info(
                "Plan iter %d/%d — critic: %s",
                iter_idx,
                max_iter,
                "approved" if approved else "needs revision",
            )
            if approved:
                break

        if draft_path is None:
            raise RuntimeError("Planner failed to produce a tutorial draft")

        return PlanResult(
            draft_json_path=draft_path,
            iterations=iterations,
            approved=approved,
            critic_feedback=critic_feedback,
        )

    def compose(
        self,
        video_path: str | Path,
        draft_json_path: str | Path,
        raw_coord_space: Optional[tuple[int, int]] = None,
    ):
        video = Path(video_path).resolve()
        draft = Path(draft_json_path).resolve()

        output_dir = self.config.composer.output_dir
        if output_dir is None:
            output_dir = draft.parent / "tutorial_img_out"

        effective_coord_space = raw_coord_space or self.config.composer.raw_coord_space
        return self.backend.compose_tutorial(
            video_path=video,
            draft_json_path=draft,
            output_dir=Path(output_dir).resolve(),
            raw_coord_space=effective_coord_space,
            sam2_config=self.config.composer.sam2,
        )

    def generate(
        self,
        session_id: str,
        video_path: str | Path,
        data_path: str | Path,
        on_stage=None,
    ) -> GenerateResult:
        video = Path(video_path).resolve()
        data = Path(data_path).resolve()
        if hasattr(self.backend, "reset_report"):
            self.backend.reset_report()

        logger.info("[1/3] Parse — extracting actions from %s", video.name)
        if on_stage:
            on_stage("parse")
        parse_result = self.parse(video, on_stage=on_stage)
        logger.info("[1/3] Parse done — trace: %s", parse_result.trace_json_path.name)

        logger.info("[2/3] Plan — generating tutorial draft")
        plan_result = self.plan(parse_result.trace_json_path, on_stage=on_stage)
        logger.info(
            "[2/3] Plan done — %d iteration(s), approved=%s",
            plan_result.iterations,
            plan_result.approved,
        )

        logger.info("[3/3] Compose — rendering tutorial")
        if on_stage:
            on_stage("compose")
        detected_video_size = _probe_video_size(video)
        detected_coord_space = (
            _read_recording_coord_space(data)
            or _infer_recording_coord_space(data, detected_video_size)
            or detected_video_size
        )
        compose_result = self.compose(
            video_path=video,
            draft_json_path=plan_result.draft_json_path,
            raw_coord_space=detected_coord_space,
        )
        logger.info("[3/3] Compose done — output: %s", compose_result.output_dir)
        report = {}
        if hasattr(self.backend, "generation_report"):
            report = self.backend.generation_report()

        return GenerateResult(
            session_id=session_id,
            video_path=video,
            data_path=data,
            parse=parse_result,
            plan=plan_result,
            compose=compose_result,
            generation_report=report,
        )
