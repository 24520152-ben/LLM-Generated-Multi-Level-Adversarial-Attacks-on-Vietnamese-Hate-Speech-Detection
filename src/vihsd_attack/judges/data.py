"""Read self-contained candidate rows and deduplicate prediction-flipping paths."""

import csv
import re
import unicodedata
from pathlib import Path

from vihsd_attack.config import LABELS, default_artifact_root

INPUT_FIELDS = ["source_id", "original", "original_label", "candidate", "path"]


def input_key(row: dict) -> tuple[str, str]:
    """Match one source and edit path across independent judge files."""
    return row["source_id"], row["path"]


def _normalize(text: str) -> str:
    """Normalize Unicode without discarding punctuation, case or diacritics."""
    return unicodedata.normalize("NFC", text)


def _inline(text: str) -> str:
    """Keep comment text on one line without confusing it with step separators."""
    return (
        text.replace("\\", "\\\\")
        .replace("\r", r"\r")
        .replace("\n", r"\n")
        .replace(" -> ", r" \-> ")
    )


def format_path(steps: list[dict]) -> str:
    """Display numbered C/W/S steps on one line, separated by ASCII arrows."""
    return " -> ".join(
        f"{i}. [{step['mode']}] {_inline(step['text'])}"
        for i, step in enumerate(steps, 1)
    )


def prepare_inputs(
    attack_dir: str | Path, output_file: str | Path | None = None
) -> Path:
    """Export only scored variants that flip their source classifier's prediction.

    Args:
        attack_dir: Parent of model folders containing results/candidates CSVs.
        output_file: Optional destination for the single portable input CSV.

    Returns:
        The five-column input CSV; identical source/path pairs share one row.

    Raises:
        ValueError: Sources, query links or labels are inconsistent.
    """
    pool, sources = {}, {}
    directories = sorted(
        path
        for path in Path(attack_dir).glob("*/candidates.csv")
        if path.parent.name != "sample"
    )
    if not directories:
        raise ValueError(f"No */candidates.csv found under {attack_dir}")
    for candidate_file in directories:
        model = candidate_file.parent.name
        histories = {}
        with candidate_file.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                source_id = row["source_id"]
                original, label = _normalize(row["original"]), row["original_label"]
                source = (original, label)
                if label not in LABELS or sources.get(source_id, source) != source:
                    raise ValueError(f"Inconsistent source/label: {source_id}")
                sources[source_id] = source
                if row["status"] != "scored":
                    continue
                context = (source_id, row["mode"], row["beam_width"])
                history = histories.setdefault(context, {row["original"]: []})
                query, parent = int(row["query"]), row["parent"]
                if (
                    query != len(history)
                    or parent not in history
                    or row["candidate"] in history
                ):
                    raise ValueError(f"Broken query path: {model}/{context}/{query}")
                if row["predicted_label"] not in LABELS or row["step_mode"] not in {
                    "C",
                    "W",
                    "S",
                }:
                    raise ValueError(
                        f"Invalid prediction/step mode: {model}/{context}/{query}"
                    )
                candidate = _normalize(row["candidate"])
                path = [
                    *history[parent],
                    {"mode": row["step_mode"], "text": candidate},
                ]
                history[row["candidate"]] = path
                if row["predicted_label"] == label:
                    continue
                readable_path = format_path(path)
                pool.setdefault(
                    (source_id, readable_path),
                    {
                        "source_id": source_id,
                        "original": original,
                        "original_label": label,
                        "candidate": candidate,
                        "path": readable_path,
                    },
                )
    if not pool:
        raise ValueError("No prediction-flipping candidates found")
    output = (
        Path(output_file)
        if output_file
        else default_artifact_root() / "judges" / "judge_inputs.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, INPUT_FIELDS)
        writer.writeheader()
        writer.writerows(item for _, item in sorted(pool.items()))
    pairs = {(item["source_id"], item["candidate"]) for item in pool.values()}
    print(
        f"Prepared {len(pairs)} flipping variants, {len(pool)} distinct paths: {output}",
        flush=True,
    )
    return output


def load_inputs(path: str | Path) -> list[dict]:
    """Validate source/path keys, numbered steps and final candidate text.

    Args:
        path: CSV produced by prepare_inputs().

    Returns:
        Unique input rows with readable, single-line edit paths.
    """
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != INPUT_FIELDS:
            raise ValueError("Expected judge_inputs.csv from prepare_inputs()")
        rows = list(reader)
    seen, sources = set(), {}
    for row in rows:
        if None in row or any(not value or not value.strip() for value in row.values()):
            raise ValueError("Incomplete judge input")
        key = input_key(row)
        parts = row["path"].split(" -> ")
        source = (row["original"], row["original_label"])
        if (
            row["original_label"] not in LABELS
            or sources.setdefault(row["source_id"], source) != source
            or "\n" in row["path"]
            or "\r" in row["path"]
            or any(
                not re.fullmatch(rf"{i}\. \[[CWS]\] .+", part)
                for i, part in enumerate(parts, 1)
            )
            or parts[-1].split("] ", 1)[-1] != _inline(row["candidate"])
            or key in seen
        ):
            raise ValueError(f"Invalid or duplicated judge input: {row['source_id']}")
        seen.add(key)
    if not rows:
        raise ValueError("Judge input CSV is empty")
    return rows
