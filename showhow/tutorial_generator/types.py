from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ParseResult:
    trace_json_path: Path
    operations_decrypted_path: Optional[Path] = None
    sequence_path: Optional[Path] = None


@dataclass
class PlanResult:
    draft_json_path: Path
    iterations: int
    approved: bool
    critic_feedback: Optional[str] = None


@dataclass
class ComposeResult:
    output_dir: Path
    markdown_path: Optional[Path] = None
    dataset_jsonl_path: Optional[Path] = None


@dataclass
class GenerateResult:
    session_id: str
    video_path: Path
    data_path: Path
    parse: ParseResult
    plan: PlanResult
    compose: ComposeResult
    generation_report: dict
