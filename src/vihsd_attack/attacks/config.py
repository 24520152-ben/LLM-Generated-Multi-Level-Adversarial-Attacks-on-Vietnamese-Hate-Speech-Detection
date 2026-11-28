"""LLM generation, beam search, and validation attack settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AttackConfig:
    """Validation settings; Q/G budgets share one trajectory per beam width."""

    checkpoint_dir: str | Path
    llm_checkpoint: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    llm_revision: str | None = None
    llm_max_memory: tuple[str, ...] = ("11GiB", "11GiB")
    classifier_device: str = "cuda:1"
    query_budgets: tuple[int, ...] = tuple(range(1, 26))
    generation_budgets: tuple[int, ...] = tuple(range(5, 26, 5))
    beam_widths: tuple[int, ...] = (1, 3, 5)
    modes: tuple[str, ...] = ("C", "W", "S", "MIX")
    sample_per_label: int = 10
    seed: int = 42
    candidates_per_call: int = 5
    max_new_tokens: int = 2048
    # Recommended sampling settings for Qwen3-30B-A3B-Instruct-2507.
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
        if not self.llm_max_memory or any(not value for value in self.llm_max_memory):
            raise ValueError("llm_max_memory must specify a memory budget per GPU")
        for name in ("query_budgets", "generation_budgets", "beam_widths"):
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
            "max_new_tokens",
            "max_char_edits",
            "max_word_spans",
            "max_span_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if any(
            budget % self.candidates_per_call
            for budget in self.generation_budgets
            if budget != max(self.generation_budgets)
        ):
            raise ValueError(
                "Intermediate generation budgets must align with candidates_per_call"
            )
        if self.seed < 0 or self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid seed or sampling settings")
        if self.top_k < 0:
            raise ValueError("top_k must be nonnegative")
