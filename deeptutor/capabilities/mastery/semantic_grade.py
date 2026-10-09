"""LLM semantic-equivalence judging for free-text mastery answers.

The deterministic grader (``deeptutor.learning.grading``) compares surface
forms: exact, fuzzy-similar, or keyword-overlapping. A Chinese learner who
answers 「两地距离是三百公里」 against the reference 「距离为300km」 is
correct and will be graded wrong every time — and a wrongly-graded attempt
feeds the error records and the spaced-repetition scheduler, so one phrasing
mismatch locks the knowledge point into endless review.

This module is the second grading layer, invoked only when the deterministic
layer already ruled the answer wrong:

* the judge is the **task** service model (falling back to the chat model) —
  the same resolution chain as answer-key verification;
* one plain completion, no tools: equivalence judgment has no arithmetic to
  delegate;
* the verdict is machine-parsed (``FINAL: AGREE / DISAGREE / UNSURE``) because
  mastery grades are permanent records that drive the scheduler, not feedback
  prose.

The public surface is one coroutine returning ``"agree"`` / ``"disagree"`` /
``"unverified"``. Callers fail **closed** on anything but ``"agree"``: judge
trouble must keep the deterministic verdict, never silently pass learners.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from deeptutor.runtime.agentic import (
    LLMClientConfig,
    build_completion_kwargs,
    build_openai_client,
)

logger = logging.getLogger(__name__)

AGREE = "agree"
DISAGREE = "disagree"
UNVERIFIED = "unverified"

JUDGE_TIMEOUT_S = 60.0
MAX_TOKENS = 1024

_FINAL_RE = re.compile(r"FINAL\s*[:：]\s*\**\s*(AGREE|DISAGREE|UNSURE)", re.IGNORECASE)

_SYSTEM_PROMPT = """You are a strict answer-equivalence judge for a learning platform.

You will see one question, its reference answer, and a learner's answer. A
deterministic grader has already ruled the learner's answer wrong on surface
form; your only job is to decide whether it is nonetheless equivalent to the
reference answer in meaning.

Judge AGREE when the learner's answer states the same answer as the
reference: paraphrase, synonyms, different word order or language, equivalent
numbers, units, or notation (0.5 = 1/2 = 50%), or — when the reference is a
list of required points — at least 60% of those points expressed in the
learner's own words.

Judge DISAGREE when it states a different concept or a different number, or
covers required points below that threshold. Never reward effort, proximity,
or answers that are merely on the same topic.

Judge UNSURE only when you genuinely cannot determine it: illegible input, an
answer unrelated to the question, or a reference answer too ambiguous to
check against.

Reply with at most two short sentences of reasoning, then end with exactly
one line:

FINAL: AGREE

(or DISAGREE / UNSURE). Output nothing after that line."""


def _messages(
    question: str, question_type: str, expected_answer: str, learner_answer: str
) -> list[dict[str, Any]]:
    if question_type == "open":
        hint = (
            "The reference answer is a list of required points "
            "(comma/semicolon separated); AGREE needs at least 60% of them."
        )
    else:
        hint = "The reference answer is the single expected short answer."
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Reference answer:\n{expected_answer}\n\n"
                f"Learner's answer:\n{learner_answer}\n\n"
                f"{hint}\nEnd with the FINAL line."
            ),
        },
    ]


def _parse_verdict(text: str) -> str:
    match = _FINAL_RE.search(text or "")
    return match.group(1).lower() if match else ""


def _network_calls_disabled() -> bool:
    """Unit tests must never spend real provider calls.

    pytest sets ``PYTEST_CURRENT_TEST`` for the duration of each test; the
    judge fails closed (UNVERIFIED) there, exactly as it does on any other
    infrastructure failure. Tests that DO want live traffic patch the public
    coroutine itself.
    """
    return "PYTEST_CURRENT_TEST" in os.environ


async def judge_free_text_answer(
    question: str,
    question_type: str,
    expected_answer: str,
    learner_answer: str,
) -> str:
    """Judge whether ``learner_answer`` means the same as ``expected_answer``.

    Any infrastructure failure maps to ``"unverified"`` so the caller can
    keep the deterministic verdict. Logs never include the answers.
    """
    if question_type not in ("short", "open"):
        return UNVERIFIED
    if not str(expected_answer or "").strip() or not str(learner_answer or "").strip():
        return UNVERIFIED
    if _network_calls_disabled():
        return UNVERIFIED
    try:
        verdict = await asyncio.wait_for(
            _judge(question, question_type, expected_answer, learner_answer),
            timeout=JUDGE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("semantic answer judging timed out after %ss", JUDGE_TIMEOUT_S)
        return UNVERIFIED
    except Exception:
        logger.warning("semantic answer judging failed", exc_info=True)
        return UNVERIFIED
    if verdict not in (AGREE, DISAGREE):
        logger.warning("semantic answer judging inconclusive: no usable FINAL line")
        return UNVERIFIED
    return verdict


async def _judge(
    question: str, question_type: str, expected_answer: str, learner_answer: str
) -> str:
    from deeptutor.services.config import resolve_llm_runtime_config
    from deeptutor.services.model_selection.runtime import llm_config_from_resolved

    resolved = resolve_llm_runtime_config(service_name="task")
    if not (getattr(resolved, "model", "") or "").strip():
        # Same fallback as answer-key verification: a task service with no
        # model selected must not ship the request unnamed to the provider.
        resolved = resolve_llm_runtime_config(service_name="llm")
    cfg = llm_config_from_resolved(resolved)
    client_config = LLMClientConfig(
        binding=getattr(cfg, "binding", None) or "openai",
        model=getattr(cfg, "model", None),
        api_key=getattr(cfg, "api_key", None),
        base_url=getattr(cfg, "effective_url", None) or getattr(cfg, "base_url", None),
        api_version=getattr(cfg, "api_version", None),
        extra_headers=getattr(cfg, "extra_headers", None) or None,
        reasoning_effort=getattr(cfg, "reasoning_effort", None),
        wire_api=getattr(cfg, "wire_api", None) or "auto",
        api_format=getattr(cfg, "api_format", None) or "auto",
    )
    client = build_openai_client(client_config)
    response = await client.chat.completions.create(
        model=client_config.model,
        messages=_messages(question, question_type, expected_answer, learner_answer),
        **build_completion_kwargs(
            temperature=0.0,
            model=client_config.model,
            max_tokens=MAX_TOKENS,
            binding=client_config.binding,
        ),
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return _parse_verdict(choices[0].message.content or "")


__all__ = ["AGREE", "DISAGREE", "UNVERIFIED", "judge_free_text_answer"]
