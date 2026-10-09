"""Deterministic answer grading + coarse error classification for Mastery Path."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re
from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from deeptutor.learning.models import ErrorType

# Verbal lead-ins that carry no answer content; longer alternatives must come
# first so they win the alternation. Bare ambiguous starters ("选"…) are
# deliberately excluded: real terminology such as 选言命题 (disjunctive
# proposition) begins with them.
_ANSWER_LEAD_RE = re.compile(
    r"^\s*(?:参考答案|应该是|大概是|大约是|我认为|我觉得|我选|final\s+answer|答案|回答|解答|应为|answer|ans)"
    r"\s*[:：是为]?\s*",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d+)?|\.\d+))?$")


def _normalize_text(value: str) -> str:
    """Fold surface forms that never change an answer's meaning.

    NFKC maps full-width characters to their ASCII halves (１２３ -> 123, ，-> ,),
    casefold removes case, verbal lead-ins ("答案是…") are dropped, and all
    whitespace and punctuation goes — 中文 answers are routinely re-punctuated
    or re-spaced with zero change in meaning. The same normalization applies
    to expected and learner answers alike, so symmetric stripping stays safe.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    for _ in range(3):  # lead-ins stack up: "我认为应该是X"
        stripped = _ANSWER_LEAD_RE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return "".join(
        ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def _numeric_value(value: str) -> float | None:
    """Parse the text as one standalone number: decimal, a/b fraction, or percent."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = text.replace(",", "").replace(" ", "")
    if not text:
        return None
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    if not _NUMBER_RE.match(text):
        return None
    try:
        numerator, _, denominator = text.partition("/")
        result = float(numerator) / float(denominator) if denominator else float(numerator)
    except (ValueError, ZeroDivisionError):
        return None
    return result / 100 if percent else result


def _bigram_similarity(a: str, b: str) -> float:
    """Dice coefficient over character bigrams — order-insensitive, unlike SequenceMatcher."""
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a == b else 0.0
    a_bigrams = Counter(a[i : i + 2] for i in range(len(a) - 1))
    b_bigrams = Counter(b[i : i + 2] for i in range(len(b) - 1))
    overlap = sum((a_bigrams & b_bigrams).values())
    return 2 * overlap / (sum(a_bigrams.values()) + sum(b_bigrams.values()))


def grade_answer(user_answer: str, expected_answer: str, question_type: str = "short") -> bool:
    """Grade user answer against expected answer.

    Args:
        user_answer: The user's submitted answer.
        expected_answer: The stored expected answer.
        question_type: One of "choice", "short", "open".

    Returns:
        True if answer is correct.
    """
    user = user_answer.strip().lower()
    expected = expected_answer.strip().lower()

    if not expected:
        return False

    if question_type == "choice":
        user_norm = user.replace(" ", "")
        expected_norm = expected.replace(" ", "")
        return user_norm == expected_norm

    if question_type == "short":
        user_norm = _normalize_text(user_answer)
        expected_norm = _normalize_text(expected_answer)
        if not user_norm or not expected_norm:
            return False
        if user_norm == expected_norm:
            return True
        user_number = _numeric_value(user_answer)
        expected_number = _numeric_value(expected_answer)
        if user_number is not None and expected_number is not None:
            return abs(user_number - expected_number) < 1e-9
        if len(expected) > 30:
            return False
        if SequenceMatcher(None, user, expected).ratio() >= 0.85:
            return True
        # A padded answer (quote the key, then keep talking) must not pass
        # just because every key bigram appears somewhere inside it.
        if len(user_norm) <= len(expected_norm) * 2 + 2:
            return _bigram_similarity(user_norm, expected_norm) >= 0.75
        return False

    if question_type == "open":
        user_norm = _normalize_text(user_answer)
        keywords = [
            keyword
            for keyword in (
                _normalize_text(part) for part in re.split(r"[,;，；。\n/]+", expected_answer)
            )
            if keyword
        ]
        if not keywords:
            return False
        matched = sum(1 for kw in keywords if kw in user_norm)
        return matched / len(keywords) >= 0.6

    return False


def classify_error(user_answer: str) -> ErrorType:
    """Coarse error classification for a wrong answer.

    A blank answer signals the student did not know (metacognitive); anything
    else is treated as a wrong application. The richer four-type taxonomy is
    assigned later by the LLM in the error-diagnosis stage.
    """
    from deeptutor.learning.models import ErrorType

    return ErrorType.METACOGNITIVE if not user_answer.strip() else ErrorType.APPLICATION_ERROR


__all__ = ["grade_answer", "classify_error"]
