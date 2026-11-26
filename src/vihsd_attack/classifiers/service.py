"""Train baselines and save the selected checkpoint and evaluation scores."""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

from vihsd_attack.classifiers.config import BaselineConfig
from vihsd_attack.classifiers.metrics import compute_metrics
from vihsd_attack.config import LABELS
from vihsd_attack.data import Example, load_vihsd


@dataclass(frozen=True)
class BaselineResult:
    output_dir: Path
    checkpoint_dir: Path
    metrics: dict


def _hf_token() -> str | None:
    """Return HF_TOKEN from the environment or Kaggle Secrets, if available."""
    if token := os.environ.get("HF_TOKEN"):
        return token
    if not Path("/kaggle/working").is_dir():
        return None
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        # The dataset may be public or the user may already be logged in.
        return None


def _evaluate(classifier, rows: list[Example], split: str) -> dict[str, float]:
    """Evaluate a split and report its metrics.

    Args:
        classifier: SVMClassifier or TransformerClassifier to evaluate.
        rows: Labeled examples from the split.
        split: Split name for the metrics output.

    Returns:
        Classification metrics for the split.
    """
    gold = [row.label for row in rows]
    predicted = []
    # Bound inference memory for both classifier types.
    for start in range(0, len(rows), 256):
        chunk = rows[start : start + 256]
        predicted.extend(classifier.predict([row.text for row in chunk]))
    metrics = compute_metrics(gold, predicted)
    print(
        f"{split}: macro-F1 = {metrics['macro_f1']:.2%}, accuracy = {metrics['accuracy']:.2%}"
    )
    return metrics


def run_baseline(config: BaselineConfig) -> BaselineResult:
    """Train a baseline and save directly under its model directory.

    Args:
        config: Model, dataset, training, and output settings for the run.

    Returns:
        The output and checkpoint directories together with validation and test
        metrics.
    """
    token = _hf_token()
    output_dir = Path(config.artifact_dir) / config.model
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    splits = load_vihsd(config, token=token)
    if config.model == "svm":
        from vihsd_attack.classifiers.svm import train_svm

        classifier = train_svm(config, splits.train)
        classifier.save(checkpoint_dir)
        (checkpoint_dir / "baseline.json").write_text(
            json.dumps({"model": "svm", "label_order": list(LABELS)}), encoding="utf-8"
        )
        best_epoch = None
        validation_history = [_evaluate(classifier, splits.validation, "validation")]
    else:
        from vihsd_attack.classifiers.transformer import train_transformer

        classifier, best_epoch, validation_history = train_transformer(
            config, splits.train, splits.validation, checkpoint_dir, token=token
        )

    validation = validation_history[(best_epoch or 1) - 1]
    test = _evaluate(classifier, splits.test, "test")
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, ["split", "epoch", "macro_f1", "accuracy"])
        writer.writeheader()
        for epoch, scores in enumerate(validation_history, start=1):
            writer.writerow(
                {
                    "split": "validation",
                    "epoch": epoch if best_epoch else None,
                    **scores,
                }
            )
        writer.writerow({"split": "test", "epoch": best_epoch, **test})

    print(f"Saved baseline: {output_dir}")
    return BaselineResult(
        output_dir, checkpoint_dir, {"validation": validation, "test": test}
    )


def load_classifier(checkpoint_dir: str | Path, *, device: str = "auto"):
    """Restore a trained classifier from a baseline checkpoint.

    Args:
        checkpoint_dir: Trusted checkpoint directory. Loading an SVM
            checkpoint unpickles its saved pipeline.
        device: Transformer inference device or "auto" to select CUDA or CPU.

    Returns:
        The fitted SVMClassifier or TransformerClassifier.

    Raises:
        ValueError: The saved label order is incompatible with ViHSD labels.
    """
    directory = Path(checkpoint_dir)
    metadata_path = directory / "baseline.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata["label_order"] != list(LABELS):
            raise ValueError("Checkpoint label order is incompatible")
        if metadata["model"] == "svm":
            from vihsd_attack.classifiers.svm import SVMClassifier

            return SVMClassifier.load(directory)
    from vihsd_attack.classifiers.transformer import TransformerClassifier

    return TransformerClassifier.load(directory, device=device)
