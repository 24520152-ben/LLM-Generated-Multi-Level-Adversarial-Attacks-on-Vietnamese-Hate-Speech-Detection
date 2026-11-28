"""Resume independent judges and combine their suggestions for human review."""

import csv
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from vihsd_attack.classifiers.service import _hf_token
from vihsd_attack.config import default_artifact_root
from vihsd_attack.judges.config import JudgeConfig
from vihsd_attack.judges.data import INPUT_FIELDS, input_key, load_inputs
from vihsd_attack.judges.generator import (
    CRITERIA,
    VOTE_FIELDS,
    LLMJudge,
)


def _review_row(source: dict, verdict: dict) -> dict:
    """Combine readable input context with one judge or consensus verdict."""
    valid = verdict["valid"]
    if valid == "NO" or verdict["label"] not in {source["original_label"], "UNSURE"}:
        valid = "NO"
    elif verdict["label"] == "UNSURE":
        valid = "UNSURE"
    return {
        **source,
        "candidate_label": verdict["label"],
        "valid": valid,
    }


def _read_votes(path: Path, inputs: dict) -> dict[tuple[str, str], dict]:
    """Read votes keyed by source/path and verify their original input context."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != VOTE_FIELDS:
            raise ValueError(f"Unexpected judge CSV schema: {path}")
        votes = {}
        for row in reader:
            if (
                None in row
                or any(value is None for value in row.values())
                or input_key(row) in votes
            ):
                raise ValueError(f"Incomplete or duplicated record in {path}")
            key = input_key(row)
            if key not in inputs or any(
                row[field] != inputs[key][field] for field in INPUT_FIELDS
            ):
                raise ValueError(f"Saved {path.name} does not match these inputs")
            # Keep the model's internal criterion name; CSVs use candidate_label.
            row["label"] = row["candidate_label"]
            votes[key] = row
    return votes


def needs_tiebreak(first: dict, second: dict) -> bool:
    """Request Gemma when either label or combined validity is unresolved.

    Args:
        first: Qwen verdict.
        second: SEA-LION verdict.

    Returns:
        Whether either result disagrees or contains UNSURE.
    """
    return any(first[key] != second[key] or first[key] == "UNSURE" for key in CRITERIA)


def _first_stage(rows: list[dict], directory: Path) -> tuple[dict, dict]:
    """Require complete first-stage results before routing inputs to Gemma."""
    inputs = {input_key(row): row for row in rows}
    votes = tuple(
        _read_votes(directory / f"{judge}.csv", inputs) for judge in ("qwen", "sealion")
    )
    if any(set(values) != inputs.keys() for values in votes):
        raise ValueError(
            "Qwen and SEA-LION must both finish this input CSV before Gemma/merge"
        )
    return votes


def run_judge(config: JudgeConfig) -> Path:
    """Append verdicts for source/path pairs not already in the judge CSV.

    Args:
        config: Input CSV, judge, device and generation settings.

    Returns:
        The seven-column judge CSV, keeping failed responses as UNSURE.

    Raises:
        ValueError: Saved cases differ from the inputs, or the first stage is incomplete.
    """
    rows = load_inputs(config.input_file)
    directory = (
        Path(config.output_dir)
        if config.output_dir
        else default_artifact_root() / "judges"
    )
    directory.mkdir(parents=True, exist_ok=True)
    if config.judge == "gemma":
        first, second = _first_stage(rows, directory)
        rows = [
            row
            for row in rows
            if needs_tiebreak(first[input_key(row)], second[input_key(row)])
        ]
    output = directory / f"{config.judge}.csv"
    done = _read_votes(output, {input_key(row): row for row in rows})
    pending = [row for row in rows if input_key(row) not in done]
    if not pending:
        if not output.exists():
            with output.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, VOTE_FIELDS).writeheader()
        print(
            f"{config.judge}: {len(done)} saved verdicts; nothing to generate",
            flush=True,
        )
        return output
    judge = LLMJudge(config, token=_hf_token())
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, VOTE_FIELDS)
        if not f.tell():
            writer.writeheader()
            f.flush()
        for row in tqdm(pending, desc=f"Judge {config.judge}"):
            writer.writerow(_review_row(row, judge.judge(row)))
            f.flush()
    print(f"Saved judge: {output}", flush=True)
    return output


def merge_judgments(
    input_file: str | Path, output_dir: str | Path | None = None
) -> Path:
    """Combine label/validity votes without discarding any human-review inputs.

    Args:
        input_file: Prepared prediction-flipping paths.
        output_dir: Directory with independent judge CSVs.

    Returns:
        Suggestions CSV with source text, source label, candidate text, edit path
        and LLM verdicts. Final human labels and valid ASR are not computed here.
    """
    rows = load_inputs(input_file)
    directory = Path(output_dir) if output_dir else default_artifact_root() / "judges"
    first, second = _first_stage(rows, directory)
    third = _read_votes(directory / "gemma.csv", {input_key(row): row for row in rows})
    expected = {
        input_key(row)
        for row in rows
        if needs_tiebreak(first[input_key(row)], second[input_key(row)])
    }
    if set(third) != expected:
        raise ValueError("Gemma must finish exactly the routed inputs before merging")
    output = directory / "suggestions.csv"
    counts = Counter()
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, VOTE_FIELDS)
        writer.writeheader()
        for row in rows:
            key = input_key(row)
            ballots = [first[key], second[key], *([third[key]] if key in third else [])]
            consensus = {}
            for criterion in CRITERIA:
                votes = Counter(
                    v[criterion] for v in ballots if v[criterion] != "UNSURE"
                )
                value, count = votes.most_common(1)[0] if votes else ("UNSURE", 0)
                consensus[criterion] = value if count >= 2 else "UNSURE"
            review = _review_row(row, consensus)
            counts[review["valid"]] += 1
            writer.writerow(review)
    print(
        f"LLM suggestions only (all still need human review): {dict(counts)}",
        flush=True,
    )
    return output
