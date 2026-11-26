"""Load official ViHSD splits while preserving source row IDs."""

from dataclasses import dataclass

from vihsd_attack.classifiers.config import BaselineConfig
from vihsd_attack.config import LABELS


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: int


@dataclass
class DatasetSplits:
    train: list[Example]
    validation: list[Example]
    test: list[Example]


def load_vihsd(config: BaselineConfig, token: str | None = None) -> DatasetSplits:
    """Load the ViHSD splits while preserving source row IDs.

    Args:
        config: Dataset name, revision, and source column settings.
        token: Optional Hugging Face access token.

    Returns:
        Train, validation, and test examples with their original row IDs.

    Raises:
        ValueError: A row has invalid text or a label ID outside the label set,
            or a split has no usable examples.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        config.dataset_name, revision=config.dataset_revision, token=token
    )
    splits = {}
    for split in ("train", "validation", "test"):
        examples = []
        for index, row in enumerate(dataset[split]):
            source_id = f"{split}:{index}"
            text = row[config.text_column]
            if text is not None and not isinstance(text, str):
                raise ValueError(f"{source_id}: text must be a string")
            if not text or not text.strip():
                continue
            label = row[config.label_column]
            if label not in range(len(LABELS)):
                raise ValueError(f"{source_id}: invalid label {label!r}")
            examples.append(Example(source_id, text, int(label)))
        if not examples:
            raise ValueError(f"No usable examples in {split}")
        splits[split] = examples
    return DatasetSplits(**splits)
