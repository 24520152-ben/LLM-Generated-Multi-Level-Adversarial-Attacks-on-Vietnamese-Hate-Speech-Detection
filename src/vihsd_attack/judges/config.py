"""Local judge checkpoints and short, deterministic generation settings."""

from dataclasses import dataclass
from pathlib import Path

JUDGE_MODELS = {
    "qwen": ("Qwen/Qwen3.5-9B", 8),
    "sealion": ("aisingapore/Llama-SEA-LION-v3.5-8B-R", 8),
    "gemma": ("google/gemma-4-12B-it", 4),
}


@dataclass(frozen=True)
class JudgeConfig:
    """Run one judge; Gemma is reserved for disagreements or uncertain answers."""

    input_file: str | Path
    judge: str = "qwen"
    device: str = "cuda:0"
    output_dir: str | Path | None = None
    max_input_tokens: int = 8192
    max_new_tokens: int = 128
    # Match the per-step limits used by the attack pilot.
    max_char_edits: int = 2
    max_word_spans: int = 2
    max_span_tokens: int = 4

    def __post_init__(self) -> None:
        """Reject unknown judges, CPU quantization and nonpositive limits."""
        if self.judge not in JUDGE_MODELS:
            raise ValueError(f"judge must be one of {tuple(JUDGE_MODELS)}")
        if not self.device.startswith("cuda:") or not self.device[5:].isdigit():
            raise ValueError("device must name one GPU, for example cuda:0")
        limits = (
            self.max_input_tokens,
            self.max_new_tokens,
            self.max_char_edits,
            self.max_word_spans,
            self.max_span_tokens,
        )
        if any(type(value) is not int or value < 1 for value in limits):
            raise ValueError("Token/edit limits must be positive")
