"""LLM generation, beam search, and validation pilot settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttackConfig:
    """Pilot settings; query budgets share one search trajectory per beam width."""

    checkpoint_dir: str | Path
    llm_checkpoint: str = "Qwen/Qwen3-8B"
    llm_revision: str | None = None
    llm_device: str = "cuda:0"
    classifier_device: str = "cuda:1"
    query_budgets: tuple[int, ...] = (10, 50, 100)
    beam_widths: tuple[int, ...] = (1, 3, 5)
    modes: tuple[str, ...] = ("C", "W", "S", "MIX")
    sample_per_label: int = 5
    seed: int = 42
    candidates_per_call: int = 5
    max_generation_slots: int = 100
    max_new_tokens: int = 2048
    # Qwen3's recommended non-thinking sampling settings.
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    # Per-step pilot limits; calibrate with manually reviewed validation examples.
    max_char_edits: int = 2
    max_word_spans: int = 2
    max_span_tokens: int = 4  # Letter/number runs, not segmented Vietnamese words.
    dataset_name: str = "sonlam1102/vihsd"
    dataset_revision: str | None = None
    output_dir: str | Path | None = None

    def __post_init__(self) -> None:
        """Reject empty grids, invalid modes, and nonpositive pilot settings."""
        for name in ("query_budgets", "beam_widths"):
            values = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or any(type(value) is not int or value < 1 for value in values)
            ):
                raise ValueError(f"{name} must contain distinct positive integers")
        if (
            not self.modes
            or len(set(self.modes)) != len(self.modes)
            or set(self.modes) - {"C", "W", "S", "MIX"}
        ):
            raise ValueError("modes must contain distinct C, W, S, or MIX values")
        for name in (
            "sample_per_label",
            "candidates_per_call",
            "max_generation_slots",
            "max_new_tokens",
            "max_char_edits",
            "max_word_spans",
            "max_span_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.seed < 0 or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid seed or sampling settings")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
