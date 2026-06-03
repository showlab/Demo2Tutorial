from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from .config import CriticConfig, PlannerConfig, SAM2Config
from .action_parser import build_compact_trace_from_events
from .online_captioner import enhance_trace_captions_online
from .quality_composer import compose_quality_tutorial
from .types import ComposeResult, ParseResult

logger = logging.getLogger(__name__)


class Demo2TutBackend:
    """Thin adapter over demo2tut modules with stable function boundaries."""

    def __init__(self, demo2tut_dir: Optional[Path] = None) -> None:
        # demo2tut_dir kept for API compatibility but no longer used
        self._last_report: dict[str, Any] = {}
        self._caption_model = "gpt-4o-mini"
        self._sam2_config: Optional[SAM2Config] = None
        self.reset_report()

    def configure_modes(
        self,
        *,
        caption_model: str,
    ) -> None:
        self._caption_model = caption_model

    def parse_actions(self, video_path: Path, on_stage: Any = None) -> ParseResult:
        session_dir = video_path.parent
        logger.info("Parse: session_dir=%s", session_dir.name)
        if not (session_dir / "events.jsonl").exists():
            raise FileNotFoundError(
                f"events.jsonl not found in session path: {session_dir}"
            )

        self._last_report["parse_mode"] = "action_parser_compact"
        parsed = build_compact_trace_from_events(session_dir)
        if on_stage:
            on_stage("caption")
        self._apply_caption_mode(parsed.trace_json_path)
        self._filter_recording_control_steps(parsed.trace_json_path)
        return parsed

    def plan_once(
        self,
        trace_json_path: Path,
        planner_cfg: PlannerConfig,
        critic_feedback: Optional[str],
        iteration_id: Optional[int],
    ) -> Path:
        from .top_down_planner import trace_to_tut_draft

        prompt_file = planner_cfg.prompt_file
        if prompt_file is None:
            prompt_file = str(Path(__file__).parent / "prompt_top_down_planning.txt")
        out_path = Path(
            trace_to_tut_draft(
                trace_file=str(trace_json_path),
                prompt_file=prompt_file,
                max_actions_for_full_processing=planner_cfg.max_actions_for_full_processing,
                max_chunk_size_for_llm=planner_cfg.max_chunk_size_for_llm,
                overlap_size=planner_cfg.overlap_size,
                merge_prompt_file=planner_cfg.merge_prompt_file,
                use_llm_merge=planner_cfg.use_llm_merge,
                max_steps_for_break=planner_cfg.max_steps_for_break,
                critic_feedback=critic_feedback,
                iter_id=iteration_id,
                model=planner_cfg.model,
            )
        )
        self._last_report["plan_mode"] = "online"
        return out_path

    def evaluate_draft(
        self, draft_json_path: Path, critic_cfg: CriticConfig, iteration_id: int
    ) -> str:
        from .tut_draft_critic import evaluate_tut_draft_llm

        self._last_report["critic_mode"] = "online"
        return evaluate_tut_draft_llm(
            draft_path=str(draft_json_path),
            model=critic_cfg.model,
            iter=iteration_id,
        )

    def compose_tutorial(
        self,
        video_path: Path,
        draft_json_path: Path,
        output_dir: Path,
        raw_coord_space: Optional[tuple[int, int]],
        sam2_config: Optional[SAM2Config] = None,
    ) -> ComposeResult:
        self._last_report["compose_mode"] = "quality"
        return self._quality_compose(
            video_path=video_path,
            draft_json_path=draft_json_path,
            output_dir=output_dir,
            raw_coord_space=raw_coord_space,
            sam2_config=sam2_config or self._sam2_config,
        )

    def _apply_caption_mode(self, trace_json_path: Path) -> None:
        try:
            enhance_trace_captions_online(
                trace_json_path=trace_json_path, model=self._caption_model
            )
            self._last_report["caption_mode"] = "online"
        except Exception:
            self._last_report["caption_mode"] = "failed"
            self._last_report["warnings"].append(
                "Screenshot-based online captioning failed."
            )
            raise

    def _filter_recording_control_steps(self, trace_json_path: Path) -> None:
        try:
            trace = json.loads(trace_json_path.read_text(encoding="utf-8"))
        except Exception:
            return
        trajectory = trace.get("trajectory")
        if not isinstance(trajectory, list):
            return

        kept = [step for step in trajectory if not _is_recording_control_step(step)]
        removed = len(trajectory) - len(kept)
        if removed <= 0:
            return

        for idx, step in enumerate(kept, start=1):
            if isinstance(step, dict):
                step["step_idx"] = idx
        trace["trajectory"] = kept
        trace_json_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._last_report["warnings"].append(
            f"Filtered {removed} ShowHow recording control step(s)."
        )
        logger.info("Parser: filtered %d recording control step(s) from trace", removed)

    def _quality_compose(
        self,
        video_path: Path,
        draft_json_path: Path,
        output_dir: Path,
        raw_coord_space: Optional[tuple[int, int]] = None,
        sam2_config: Optional[SAM2Config] = None,
    ) -> ComposeResult:
        out_dir, markdown, dataset = compose_quality_tutorial(
            video_path=video_path,
            draft_json_path=draft_json_path,
            output_dir=output_dir,
            raw_coord_space=raw_coord_space,
            sam2_config=sam2_config,
        )
        return ComposeResult(
            output_dir=out_dir, markdown_path=markdown, dataset_jsonl_path=dataset
        )

    def generation_report(self) -> dict[str, Any]:
        return dict(self._last_report)

    def reset_report(self) -> None:
        self._last_report = {
            "effective_mode": "online",
            "caption_mode": "online",
            "parse_mode": None,
            "plan_mode": None,
            "critic_mode": "online",
            "compose_mode": None,
            "fallback_events": [],
            "warnings": [],
        }


def is_critic_approved(critic_output: str) -> bool:
    txt = (critic_output or "").strip().lower()
    if txt in {"all good", "looks good.", "looks good"}:
        return True
    try:
        parsed = json.loads(critic_output)
        scores = parsed.get("scores", {})
        return (
            float(scores.get("overall", 0)) >= 0.86
            and float(scores.get("coverage", 0)) >= 0.85
            and float(scores.get("granularity", 0)) >= 0.80
            and float(scores.get("orderliness", 0)) >= 0.85
            and float(scores.get("learnability", 0)) >= 0.80
        )
    except Exception:
        return False


_RECORDING_CONTROL_RE = re.compile(
    r"("
    r"\b(start|begin|stop|end|finish)\s+(the\s+)?(screen\s+)?recording\b|"
    r"\b(start|stop)\s+record\b|"
    r"\b(record|recording)\s+(button|control)\b|"
    r"(开始|开启|停止|停下|结束).{0,8}(录制|记录)|"
    r"(录制|记录).{0,8}(开始|开启|停止|停下|结束)"
    r")",
    re.IGNORECASE,
)


def _is_recording_control_step(step: dict[str, Any]) -> bool:
    if not isinstance(step, dict):
        return False
    caption = step.get("caption") or {}
    if not isinstance(caption, dict):
        return False
    action = str(caption.get("action") or "").strip()
    if not action:
        return False
    return bool(_RECORDING_CONTROL_RE.search(action))
