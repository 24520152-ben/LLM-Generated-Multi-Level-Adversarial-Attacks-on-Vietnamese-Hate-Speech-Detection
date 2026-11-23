"""Accuracy and macro-F1 for the fixed ViHSD label set."""

from sklearn.metrics import accuracy_score, f1_score

from vihsd_attack.config import LABELS


def compute_metrics(gold: list[int], predicted: list[int]) -> dict[str, float]:
    """Compute accuracy and macro-F1 across all three ViHSD labels.

    Args:
        gold: Ground-truth label IDs.
        predicted: Predicted label IDs in the same example order as gold.

    Returns:
        Accuracy and macro-F1 on the 0–1 scale.

    Raises:
        ValueError: The inputs are empty or have different lengths.
    """
    if not gold or len(gold) != len(predicted):
        raise ValueError("Gold and predictions must be nonempty and aligned")
    labels = list(range(len(LABELS)))
    return {
        "macro_f1": float(
            f1_score(gold, predicted, labels=labels, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(gold, predicted)),
    }
