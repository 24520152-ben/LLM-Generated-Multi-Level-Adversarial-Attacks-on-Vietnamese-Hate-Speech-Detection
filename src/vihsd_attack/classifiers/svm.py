"""Character TF-IDF + Linear SVM, including the vectorizer in the checkpoint."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from vihsd_attack.classifiers.config import BaselineConfig
from vihsd_attack.config import LABELS
from vihsd_attack.data import Example


class SVMClassifier:
    score_type = "decision_score"

    def __init__(self, pipeline):
        """Wrap a fitted SVM text classifier.

        Args:
            pipeline: Scikit-learn Pipeline with character TF-IDF and LinearSVC.
        """
        self.pipeline = pipeline

    def predict_scores(self, texts: list[str]) -> list[list[float]]:
        """Score texts with the fitted SVM.

        Args:
            texts: Raw input texts in prediction order.

        Returns:
            One list of decision scores per text, in ViHSD label order.
        """
        if not texts:
            return []
        classes = list(self.pipeline.classes_)
        scores = self.pipeline.decision_function(texts)
        return scores[:, [classes.index(i) for i in range(len(LABELS))]].tolist()

    def predict(self, texts: list[str]) -> list[int]:
        """Return one predicted ViHSD label ID per text."""
        if not texts:
            return []
        return self.pipeline.predict(texts).tolist()

    def save(self, directory: Path) -> None:
        """Save the fitted vectorizer and classifier together in model.joblib."""
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, directory / "model.joblib")

    @classmethod
    def load(cls, directory: Path):
        """Restore a fitted SVM classifier.

        Args:
            directory: Checkpoint directory containing a trusted model.joblib.

        Returns:
            An SVMClassifier wrapping the restored pipeline.
        """
        return cls(joblib.load(Path(directory) / "model.joblib"))


def train_svm(config: BaselineConfig, rows: list[Example]) -> SVMClassifier:
    """Fit a character TF-IDF and LinearSVC pipeline.

    Args:
        config: Vectorizer, SVM, and seed settings.
        rows: Labeled training examples.

    Returns:
        The fitted SVM classifier, including its vectorizer.

    Raises:
        ValueError: The examples do not cover exactly the three ViHSD labels.
    """
    if {row.label for row in rows} != set(range(len(LABELS))):
        raise ValueError("SVM training requires all three labels")
    pipeline = make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=config.char_ngram_range,
            min_df=config.min_df,
        ),
        LinearSVC(
            C=config.svm_c,
            class_weight=config.svm_class_weight,
            random_state=config.seed,
        ),
    )
    pipeline.fit([row.text for row in rows], [row.label for row in rows])
    return SVMClassifier(pipeline)
