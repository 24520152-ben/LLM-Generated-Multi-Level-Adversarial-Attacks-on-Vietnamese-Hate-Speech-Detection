"""Labels and artifact location shared by classifiers and attacks."""

from pathlib import Path

LABELS = ("CLEAN", "OFFENSIVE", "HATE")


def default_artifact_root() -> Path:
    """Return the shared artifact root under Kaggle or the current directory."""
    root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
    return root / "artifacts"
