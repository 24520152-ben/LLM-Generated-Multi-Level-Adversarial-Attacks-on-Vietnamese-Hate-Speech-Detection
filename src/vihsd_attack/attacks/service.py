"""Run a validation-only attack grid and save two CSV files for review."""

import csv
import random
from collections import defaultdict
from itertools import product
from pathlib import Path

from tqdm.auto import tqdm

from vihsd_attack.attacks.config import AttackConfig
from vihsd_attack.attacks.generator import CandidateGenerator
from vihsd_attack.attacks.search import beam_search
from vihsd_attack.classifiers.config import BaselineConfig
from vihsd_attack.classifiers.service import _hf_token, load_classifier
from vihsd_attack.config import LABELS, default_artifact_root
from vihsd_attack.data import load_vihsd


def run_pilot(config: AttackConfig) -> Path:
    """Evaluate beam widths on one stratified set of eligible validation rows.

    Args:
        config: Checkpoint paths, devices, query grid, and generation settings.

    Returns:
        Output directory containing candidates.csv and results.csv.

    Raises:
        ValueError: A validation label has too few eligible examples.
    """
    token = _hf_token()
    classifier = load_classifier(config.checkpoint_dir, device=config.classifier_device)
    splits = load_vihsd(
        BaselineConfig(
            dataset_name=config.dataset_name,
            dataset_revision=config.dataset_revision,
        ),
        token=token,
    )
    eligible = {label: [] for label in range(len(LABELS))}
    for start in tqdm(
        range(0, len(splits.validation), 128), desc="Validation eligibility"
    ):
        chunk = splits.validation[start : start + 128]
        scores = classifier.predict_scores([row.text for row in chunk])
        for row, values in zip(chunk, scores, strict=True):
            if max(range(len(values)), key=values.__getitem__) == row.label:
                eligible[row.label].append((row, values))
    rng = random.Random(config.seed)
    sample = []
    for label, rows in eligible.items():
        if len(rows) < config.sample_per_label:
            raise ValueError(
                f"{LABELS[label]}: only {len(rows)} eligible rows; lower sample_per_label"
            )
        sample.extend(rng.sample(rows, config.sample_per_label))
    rng.shuffle(sample)
    counts = {LABELS[label]: len(rows) for label, rows in eligible.items()}
    print(
        f"Pilot: {len(sample)} validation rows; eligible counts: {counts}", flush=True
    )
    generator = CandidateGenerator(config, token=token)
    output_dir = (
        Path(config.output_dir)
        if config.output_dir
        else default_artifact_root() / "attacks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    context_fields = ["id", "mode", "beam_width"]
    candidate_fields = context_fields + [
        "query",
        "parent_query",
        "step_mode",
        "candidate",
        "prediction",
        "status",
    ]
    result_fields = context_fields + [
        "original",
        "label",
        "query_budget",
        "generation_budget",
        "queries_used",
        "auto_success",
        "llm_calls",
        "seconds",
    ]
    print(f"Saving pilot to {output_dir}", flush=True)
    total = len(sample) * len(config.modes) * len(config.beam_widths)
    successes = defaultdict(int)
    with (
        (output_dir / "candidates.csv").open(
            "w", encoding="utf-8", newline=""
        ) as candidates,
        (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as results,
        tqdm(total=total, desc="Attack searches") as progress,
    ):
        candidate_writer = csv.DictWriter(candidates, candidate_fields)
        result_writer = csv.DictWriter(results, result_fields)
        candidate_writer.writeheader()
        result_writer.writeheader()
        for (example, scores), mode, width in product(
            sample, config.modes, config.beam_widths
        ):
            context = {
                "id": example.id,
                "mode": mode,
                "beam_width": width,
            }
            progress.set_postfix(id=example.id, mode=mode, beam=width)
            events, snapshots = beam_search(
                example,
                scores,
                classifier,
                generator,
                config,
                mode,
                width,
            )
            candidate_writer.writerows({**context, **row} for row in events)
            source = {
                **context,
                "original": example.text,
                "label": LABELS[example.label],
            }
            result_writer.writerows({**source, **row} for row in snapshots)
            successes[mode, width] += snapshots[-1]["auto_success"]
            candidates.flush()
            results.flush()
            progress.update()
    for (mode, width), count in sorted(successes.items()):
        print(
            f"{mode} beam={width} Q={max(config.query_budgets)} "
            f"G={max(config.generation_budgets)}: automatic ASR "
            f"{count / len(sample):.2%} ({count}/{len(sample)})",
            flush=True,
        )
    print(f"Saved pilot: {output_dir}", flush=True)
    return output_dir
