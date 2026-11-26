"""Budget-independent beam trajectories with per-query pilot records."""

import hashlib
import math
from dataclasses import dataclass
from itertools import product
from time import perf_counter

from vihsd_attack.attacks.config import AttackConfig
from vihsd_attack.attacks.filters import mode_matches
from vihsd_attack.config import LABELS
from vihsd_attack.data import Example


@dataclass
class Candidate:
    """One beam state, referenced by its classifier query number."""

    text: str
    margin: float
    query: int = 0


def margin(scores: list[float], label: int) -> float:
    """Return the highest competing class score minus the source class score."""
    if len(scores) != len(LABELS) or not all(math.isfinite(x) for x in scores):
        raise ValueError("Classifier must return three finite scores")
    return max(score for i, score in enumerate(scores) if i != label) - scores[label]


def beam_search(
    example: Example,
    initial_scores: list[float],
    classifier,
    generator,
    config: AttackConfig,
    mode: str,
    beam_width: int,
) -> tuple[list[dict], list[dict]]:
    """Search once and read each query-budget result from the same trajectory.

    Args:
        example: Correctly classified source example.
        initial_scores: Eligibility scores, obtained outside the attack budget.
        classifier: Loaded baseline with predict_scores().
        generator: Candidate generator with generate().
        config: Budgets, structural screens, and sampling settings.
        mode: C, W, S, or MIX for this search.
        beam_width: Number of states retained after each round.

    Returns:
        Candidate audit rows and per-budget result rows. Success is automatic;
        semantic and label validity must be assessed separately by annotators.
    """
    if mode not in {"C", "W", "S", "MIX"} or beam_width < 1:
        raise ValueError("Invalid mode or beam width")
    if max(range(len(initial_scores)), key=initial_scores.__getitem__) != example.label:
        raise ValueError("Only correctly classified examples can be attacked")
    original = Candidate(example.text, margin(initial_scores, example.label))
    beam, seen = [original], {example.text}
    events, scored, calls_at_query = [], [], []
    queries = calls = round_id = slots_used = 0
    seed_key = f"{config.seed}:{example.id}:{mode}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:4], "big")
    max_queries = max(config.query_budgets)
    started = perf_counter()
    while beam and queries < max_queries and slots_used < config.max_generation_slots:
        children = []
        # Rotate MIX's starting mode across examples and rounds, independent of Q.
        modes = [mode] if mode != "MIX" else ["C", "W", "S"]
        offset = (seed + round_id) % len(modes)
        modes = modes[offset:] + modes[:offset]
        for parent, step_mode in product(beam, modes):
            count = min(
                config.candidates_per_call, config.max_generation_slots - slots_used
            )
            if queries >= max_queries or count == 0:
                break
            calls += 1
            # Reserve requested slots even when generation fails or is filtered.
            slots_used += count
            generated = generator.generate(
                example.text,
                parent.text,
                example.label,
                step_mode,
                (seed + calls) % (2**32),
                count,
            )
            pending = []
            proposals = generated.candidates
            slots = proposals + [None] * max(0, count - len(proposals))
            for slot, value in enumerate(slots):
                text = value.strip() if isinstance(value, str) else ""
                row = {
                    "step_mode": step_mode,
                    "parent_query": parent.query,
                    "candidate": text,
                    "query": "",
                    "prediction": "",
                }
                if slot >= count:
                    status = "extra"
                elif slot >= len(proposals):
                    status = "missing"
                    if slot == 0 and generated.raw:
                        status = "invalid_output"
                        row["candidate"] = generated.raw
                elif not text:
                    status = "format"
                elif not mode_matches(parent.text, text, step_mode, config):
                    status = "mode"
                else:
                    status = "duplicate" if text in seen else "unscored"
                    if status == "unscored":
                        seen.add(text)
                        pending.append(row)
                row["status"] = status
                events.append(row)
            pending = pending[: max_queries - queries]
            if not pending:
                continue
            scores = classifier.predict_scores([row["candidate"] for row in pending])
            for row, values in zip(pending, scores, strict=True):
                queries += 1
                value = margin(values, example.label)
                predicted = max(range(len(values)), key=values.__getitem__)
                row.update(
                    status="scored",
                    query=queries,
                    prediction=LABELS[predicted],
                )
                children.append(Candidate(row["candidate"], value, queries))
                scored.append(row)
                calls_at_query.append(calls)
        # Stable sorting preserves proposal order when scores tie.
        beam = sorted(children, key=lambda item: item.margin, reverse=True)[:beam_width]
        round_id += 1
    elapsed = perf_counter() - started
    results = []
    for budget in sorted(config.query_budgets):
        prefix = scored[:budget]
        exhausted = queries < budget
        results.append(
            {
                "query_budget": budget,
                "queries_used": len(prefix),
                "auto_success": any(
                    row["prediction"] != LABELS[example.label] for row in prefix
                ),
                "llm_calls": calls if exhausted else calls_at_query[len(prefix) - 1],
                # Full-trajectory runtime appears only once, at the largest budget.
                "seconds": round(elapsed, 3) if budget == max_queries else "",
            }
        )
    return events, results
