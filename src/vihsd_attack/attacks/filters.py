"""Structural pilot screens; meaning and label preservation need human review."""

import unicodedata
from difflib import SequenceMatcher

import regex
from rapidfuzz.distance import Levenshtein

from vihsd_attack.attacks.config import AttackConfig


def mode_matches(parent: str, candidate: str, mode: str, config: AttackConfig) -> bool:
    """Screen the current edit without certifying its linguistic level or meaning.

    Args:
        parent: Text before this step.
        candidate: Proposed text after this step.
        mode: Requested C, W, or S mode.
        config: Per-step character, span-count, and span-length limits.

    Returns:
        Whether the edit passes the structural screen. W uses bounded spans of
        letter/number runs; S checks changed lexical content only. Annotators
        must confirm the requested level, meaning, and label preservation.

    Raises:
        ValueError: The step mode is not C, W, or S.
    """
    if mode not in {"C", "W", "S"}:
        raise ValueError(f"Unsupported step mode: {mode}")
    parent, candidate = (
        unicodedata.normalize("NFC", text) for text in (parent, candidate)
    )
    if not parent.strip() or not candidate.strip() or parent == candidate:
        return False
    if mode == "C":
        return (
            Levenshtein.distance(
                regex.findall(r"\X", parent),
                regex.findall(r"\X", candidate),
                score_cutoff=config.max_char_edits,
            )
            <= config.max_char_edits
        )

    # These surface units are explicit and avoid assuming correct word segmentation.
    before, after = (
        regex.findall(r"[\p{L}\p{N}]+", text) for text in (parent, candidate)
    )
    if not after or "".join(before).casefold() == "".join(after).casefold():
        return False
    if mode == "S":
        return True  # Sentence-level paraphrasing is assessed during annotation.

    edits = [
        op
        for op in SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
        if op[0] != "equal"
    ]
    return 0 < len(edits) <= config.max_word_spans and all(
        max(end - start, new_end - new_start) <= config.max_span_tokens
        for _, start, end, new_start, new_end in edits
    )
