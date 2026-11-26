"""Shared encoder training and inference with persisted preprocessing settings."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from vihsd_attack.classifiers.config import BaselineConfig
from vihsd_attack.classifiers.metrics import compute_metrics
from vihsd_attack.config import LABELS
from vihsd_attack.data import Example


def resolve_device(device: str) -> str:
    """Resolve "auto" to CUDA if available, otherwise CPU."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class TransformerClassifier:
    score_type = "logits"

    def __init__(
        self,
        model,
        tokenizer,
        *,
        segmentation: str,
        max_length: int,
        batch_size: int = 16,
        device: str = "auto",
    ):
        """Place the model on its device and configure text preprocessing.

        Args:
            model: Hugging Face model for sequence classification.
            tokenizer: Hugging Face tokenizer paired with the model.
            segmentation: "underthesea" for word segmentation or "none".
            max_length: Maximum tokenized sequence length.
            batch_size: Number of texts processed per inference batch.
            device: PyTorch device or "auto" to select CUDA or CPU.

        Raises:
            ValueError: The segmentation mode is unsupported.
        """
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.segmentation = segmentation
        self.max_length = max_length
        self.batch_size = batch_size
        if segmentation == "underthesea":
            from underthesea import word_tokenize

            self._segment = lambda text: word_tokenize(text, format="text")
        elif segmentation == "none":
            self._segment = lambda text: text
        else:
            raise ValueError(f"Unsupported segmentation: {segmentation}")

    def encode(self, texts: list[str], *, padding: bool = True):
        """Tokenize texts with the configured segmentation and token limit.

        Args:
            texts: Raw input texts.
            padding: Pad to the longest input and return tensors when true.

        Returns:
            A BatchEncoding with padded tensors or unpadded token lists.
        """
        return self.tokenizer(
            [self._segment(text) for text in texts],
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt" if padding else None,
        )

    def predict_scores(self, texts: list[str]) -> list[list[float]]:
        """Score texts with the Transformer in inference mode.

        Args:
            texts: Raw input texts in prediction order.

        Returns:
            One list of logits per text, in ViHSD label order.
        """
        self.model.eval()
        scores = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = self.encode(texts[start : start + self.batch_size])
                logits = self.model(**batch.to(self.device)).logits
                scores.extend(logits.float().cpu().tolist())
        return scores

    def predict(self, texts: list[str]) -> list[int]:
        """Return the label ID with the highest logit for each text."""
        return [int(np.argmax(row)) for row in self.predict_scores(texts)]

    def save(self, directory: Path) -> None:
        """Save the model, tokenizer, and preprocessing settings together."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        settings = {
            "segmentation": self.segmentation,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "label_order": list(LABELS),
        }
        (directory / "preprocessing.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(
        cls, directory: Path, *, device: str = "auto", batch_size: int | None = None
    ):
        """Restore the Transformer and its saved preprocessing.

        Args:
            directory: Directory containing the model and preprocessing files.
            device: PyTorch device or "auto" to select CUDA or CPU.
            batch_size: Optional override for the saved inference batch size.

        Returns:
            A TransformerClassifier ready for inference.

        Raises:
            ValueError: The saved label order is incompatible with ViHSD labels.
        """
        directory = Path(directory)
        settings = json.loads((directory / "preprocessing.json").read_text())
        if settings["label_order"] != list(LABELS):
            raise ValueError("Checkpoint label order is incompatible")
        return cls(
            AutoModelForSequenceClassification.from_pretrained(directory),
            AutoTokenizer.from_pretrained(directory),
            segmentation=settings["segmentation"],
            max_length=settings["max_length"],
            batch_size=batch_size or settings["batch_size"],
            device=device,
        )


def train_transformer(
    config: BaselineConfig,
    train: list[Example],
    validation: list[Example],
    checkpoint_dir: Path,
    token: str | None = None,
) -> tuple[TransformerClassifier, int, list[dict[str, float]]]:
    """Fine-tune a Transformer and select the checkpoint by validation macro-F1.

    Args:
        config: Model, training, and device settings.
        train: Labeled examples used for fitting.
        validation: Labeled examples scored after each epoch.
        checkpoint_dir: Destination for the selected checkpoint.
        token: Optional Hugging Face access token for the pretrained model.

    Returns:
        The reloaded classifier, best epoch, and validation metrics by epoch.

    Raises:
        ValueError: The configured token limit exceeds checkpoint support.
        RuntimeError: The training loss is non-finite.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    checkpoint = config.resolved_checkpoint
    pretrained_options = {"revision": config.model_revision, "token": token}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **pretrained_options)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={name: i for i, name in enumerate(LABELS)},
        **pretrained_options,
    )
    max_positions = getattr(model.config, "max_position_embeddings", None)
    # RoBERTa-style position IDs start after the padding index.
    position_limit = (
        max_positions - (model.config.pad_token_id or 0) - 1
        if max_positions is not None
        else config.max_length
    )
    if config.max_length > min(tokenizer.model_max_length, position_limit):
        raise ValueError(
            "max_length exceeds this checkpoint's supported sequence length"
        )
    classifier = TransformerClassifier(
        model,
        tokenizer,
        segmentation="underthesea" if config.model == "phobert" else "none",
        max_length=config.max_length,
        batch_size=config.batch_size,
        device=config.device,
    )
    print(f"Training {config.model} on {classifier.device}")
    # Tokenize once; dynamic padding still happens per batch.
    encodings = classifier.encode([row.text for row in train], padding=False)
    training_rows = [
        {
            **{key: values[i] for key, values in encodings.items()},
            "label": int(row.label),
        }
        for i, row in enumerate(train)
    ]
    loader = DataLoader(
        training_rows,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    validation_texts = [row.text for row in validation]
    validation_labels = [row.label for row in validation]
    best_score, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in tqdm(loader, desc=f"{config.model} epoch {epoch}/{config.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch.to(classifier.device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        predictions = classifier.predict(validation_texts)
        metrics = compute_metrics(validation_labels, predictions)
        history.append(metrics)
        score = metrics["macro_f1"]
        print(f"Epoch {epoch}: validation macro-F1 = {score:.2%}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            classifier.save(checkpoint_dir)
    # Release optimizer state before reloading the selected model.
    del optimizer, classifier, model, loss, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    selected = TransformerClassifier.load(checkpoint_dir, device=config.device)
    return selected, best_epoch, history
