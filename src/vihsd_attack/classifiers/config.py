"""Baseline model presets and training settings."""

from dataclasses import dataclass, field
from pathlib import Path

from vihsd_attack.config import default_artifact_root

CHECKPOINTS = {
    "phobert": "vinai/phobert-base",
    "visobert": "uitnlp/visobert",
    "xlmr": "FacebookAI/xlm-roberta-base",
}


@dataclass(frozen=True)
class BaselineConfig:
    model: str = "phobert"
    seed: int = 42
    dataset_name: str = "sonlam1102/vihsd"
    dataset_revision: str | None = None
    text_column: str = "free_text"
    label_column: str = "label_id"
    artifact_dir: Path = field(
        default_factory=lambda: default_artifact_root() / "baselines"
    )
    # None selects the preset checkpoint for this model.
    checkpoint: str | None = None
    model_revision: str | None = None
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_length: int = 256
    max_grad_norm: float = 1.0
    device: str = "auto"  # Also accepts "cpu", "cuda:0", or "cuda:1".
    svm_c: float = 1.0
    svm_class_weight: str | None = "balanced"
    # Zhang et al. (SemEval 2019) evaluated this character n-gram range.
    # https://aclanthology.org/S19-2078/
    char_ngram_range: tuple[int, int] = (2, 5)
    min_df: int = 1

    def __post_init__(self) -> None:
        """Validate the baseline settings after initialization.

        Raises:
            ValueError: The model or training settings are invalid.
        """
        if self.model not in ("svm", *CHECKPOINTS):
            raise ValueError(f"Unknown model: {self.model}")
        for name in ("batch_size", "epochs", "max_length", "min_df"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or self.svm_c <= 0 or self.max_grad_norm <= 0:
            raise ValueError("Learning rate, SVM C, and gradient norm must be positive")
        if self.weight_decay < 0 or self.seed < 0:
            raise ValueError("Weight decay and seed must be nonnegative")
        low, high = self.char_ngram_range
        if low < 1 or high < low:
            raise ValueError("Invalid character ngram range")
        if self.svm_class_weight not in (None, "balanced"):
            raise ValueError("svm_class_weight must be None or 'balanced'")

    @property
    def resolved_checkpoint(self) -> str:
        """The explicit checkpoint or the selected model's preset."""
        return self.checkpoint or CHECKPOINTS[self.model]
