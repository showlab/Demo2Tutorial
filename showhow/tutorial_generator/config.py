from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _read_demo2tut_coord_space() -> Optional[tuple[int, int]]:
    """No static default; coord space is detected at runtime from the video."""
    return None


@dataclass
class PlannerConfig:
    model: str = "gpt-4o-mini"
    prompt_file: Optional[str] = None
    max_actions_for_full_processing: int = 200
    max_chunk_size_for_llm: int = 50
    overlap_size: int = 5
    merge_prompt_file: Optional[str] = None
    use_llm_merge: bool = False
    max_steps_for_break: int = 10


@dataclass
class CriticConfig:
    enabled: bool = True
    max_iterations: int = 4
    model: str = "gpt-4.1"


def _sam2_root() -> Path:
    return Path(__file__).resolve().parents[2]  # ShowHow/


@dataclass
class SAM2Config:
    enabled: bool = False
    checkpoint: str = str(_sam2_root() / "sam2.1_hiera_tiny.pt")
    config_yaml: str = str(_sam2_root() / "sam2.1_hiera_t.yaml")
    device: str = "cpu"


@dataclass
class ComposerConfig:
    output_dir: Optional[Path] = None
    raw_coord_space: Optional[tuple[int, int]] = field(
        default_factory=_read_demo2tut_coord_space
    )
    sam2: SAM2Config = field(default_factory=SAM2Config)


@dataclass
class GeneratorConfig:
    caption_model: str = "gpt-4o"
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    composer: ComposerConfig = field(default_factory=ComposerConfig)
