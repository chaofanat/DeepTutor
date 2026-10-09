"""Independent answer-key verification for ``mastery_quiz`` registrations.

The tutor registers a choice question's ``expected_answer`` at authoring
time, and the deterministic grader trusts it forever after. Three
consecutive live mis-registrations — each a constraint-enumeration or
counting slip the tutor made while composing the question — showed that a
wrong key silently grades a correct learner as incorrect until the tutor
happens to notice and void the attempt.

This module re-solves the question with an independent agent before the
question is committed:

* the model is the **task** service (falling back to the chat model) — never
  told which answer the tutor registered, so it cannot anchor on it;
* it must do every calculation and enumeration through the sandboxed
  ``exec`` tool (the same contract the chat surface exposes) instead of
  reasoning out arithmetic in its head;
* the registered key is compared against the verifier's letter only after
  the loop finishes.

The public surface is one coroutine returning ``"agree"`` / ``"disagree"``
/ ``"unverified"``. Callers fail open on ``"unverified"``: infrastructure
trouble must not block quiz posing, only a confident disagreement may.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from deeptutor.core.tool_protocol import ToolResult
from deeptutor.runtime.agentic import (
    LLMClientConfig,
    build_completion_kwargs,
    build_openai_client,
    can_use_native_tool_calling,
)
from deeptutor.tools.exec_tool import ExecTool

logger = logging.getLogger(__name__)

AGREE = "agree"
DISAGREE = "disagree"
UNVERIFIED = "unverified"

# The verifier needs one tool round to compute and one to answer; six covers
# a checked re-solve without letting a looping model stall registration.
MAX_ROUNDS = 6
TOTAL_TIMEOUT_S = 150.0
PER_ROUND_MAX_TOKENS = 2048
# Tool output fed back into the loop is capped so a chatty program cannot
# crowd out the question itself.
TOOL_RESULT_CHAR_CAP = 4000

_FINAL_RE = re.compile(r"FINAL\s*[:：]\s*\**\s*([A-Za-z]+)")
_UNSURE_RE = re.compile(r"FINAL\s*[:：]\s*\**\s*UNSURE", re.IGNORECASE)
_SHORT_FINAL_RE = re.compile(r"FINAL\s*[:：]\s*(.+)")

_SYSTEM_PROMPT = """You are an independent answer-key verifier for a multiple-choice question.

Solve the question rigorously, on your own. You are NOT told which option
was registered as correct — judge only from the question text.

You have the `exec` tool: run Python code for EVERY calculation and
enumeration. Never do arithmetic or counting in your head. When the question
involves counting candidates, write a short program that enumerates them and
checks each against the stated constraints, then count with the program.

When you are certain, end your reply with exactly one line:

FINAL: <letter of the single correct option>

If the question is ambiguous, has zero or several correct options, or you
cannot determine the answer, end with:

FINAL: UNSURE

Output nothing after that line."""

_SHORT_SYSTEM_PROMPT = """You are an independent answer verifier for a short-answer question.

Solve the question rigorously, on your own. You are NOT told the registered
answer — judge only from the question text.

You have the `exec` tool: run Python code for EVERY calculation and
enumeration. Never do arithmetic or counting in your head. When the question
involves counting candidates, write a short program that enumerates them and
checks each against the stated constraints, then count with the program.

When you are certain, end your reply with exactly one line:

FINAL: <the answer in its minimal form — a single number, term, or short phrase>

If the question is ambiguous, has several defensible answers, or you cannot
determine it, end with:

FINAL: UNSURE

Output nothing after that line."""


def _messages(question: str, options: list[dict[str, str]]) -> list[dict[str, Any]]:
    lines = [f"{opt['label']}. {opt['body']}" for opt in options]
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\nOptions:\n"
                + "\n".join(lines)
                + "\n\nSolve it (use exec for all arithmetic), then end with the FINAL line."
            ),
        },
    ]


def _parse_final(text: str, valid_labels: set[str]) -> str:
    """Extract the verifier's FINAL letter; return "" when unusable."""
    match = _FINAL_RE.search(text or "")
    if not match:
        return ""
    token = match.group(1).upper()
    if _UNSURE_RE.search(text or "") and token == "UNSURE":
        return ""
    return token if token in valid_labels else ""


async def _run_tool_loop(
    client: Any,
    model: str | None,
    binding: str | None,
    messages: list[dict[str, Any]],
    tool_schema: dict[str, Any],
    exec_tool: ExecTool,
) -> str:
    """Native-tool-calling rounds until the model replies without tools."""
    completion_kwargs = build_completion_kwargs(
        temperature=0.0,
        model=model,
        max_tokens=PER_ROUND_MAX_TOKENS,
        binding=binding,
    )
    for round_idx in range(MAX_ROUNDS):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **completion_kwargs,
        }
        if round_idx < MAX_ROUNDS - 1:
            kwargs["tools"] = [tool_schema]
            kwargs["tool_choice"] = "auto"
        response = await client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        message = choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            return message.content or ""
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments or "{}",
                        },
                    }
                    for call in tool_calls
                ],
            }
        )
        for call in tool_calls:
            content = await _execute(exec_tool, call)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return ""


async def _execute(exec_tool: ExecTool, call: Any) -> str:
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except (TypeError, ValueError):
        return "invalid tool arguments: pass a JSON object of exec parameters"
    try:
        result: ToolResult = await exec_tool.execute(**arguments)
    except Exception as exc:  # a failed run is feedback, not a crash
        logger.warning("verifier exec failed: %s", exc)
        return f"execution error: {exc}"
    content = result.content or ""
    return content[:TOOL_RESULT_CHAR_CAP]


async def _run_without_tools(
    client: Any, model: str | None, binding: str | None, messages: list[dict[str, Any]]
) -> str:
    """Fallback for providers without native tool calling: one plain call."""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        **build_completion_kwargs(
            temperature=0.0, model=model, max_tokens=PER_ROUND_MAX_TOKENS, binding=binding
        ),
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return choices[0].message.content or ""


def _network_calls_disabled() -> bool:
    """Unit tests must never spend real provider calls.

    pytest sets ``PYTEST_CURRENT_TEST`` for the duration of each test. The
    suite (and CI) has always assumed verification is inert under pytest —
    tests with saved local credentials would otherwise fire live requests,
    and a factually-wrong registered key would be rightly rejected mid-test.
    Tests that DO want live traffic patch the public coroutine itself.
    """
    return "PYTEST_CURRENT_TEST" in os.environ


async def verify_answer_key(
    question: str,
    options: list[dict[str, str]],
    expected_label: str,
) -> str:
    """Re-solve ``question`` independently and compare with ``expected_label``.

    The verifier never sees ``expected_label``; only the returned verdict
    reflects it. Any infrastructure failure maps to ``"unverified"`` so the
    caller can fail open.
    """
    if not options or not expected_label:
        return UNVERIFIED
    if _network_calls_disabled():
        return UNVERIFIED
    valid_labels = {str(opt.get("label") or "").strip().upper() for opt in options}
    valid_labels.discard("")
    if expected_label.strip().upper() not in valid_labels:
        return UNVERIFIED
    try:
        verdict = await asyncio.wait_for(
            _verify(question, options, valid_labels, expected_label),
            timeout=TOTAL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("answer-key verification timed out after %ss", TOTAL_TIMEOUT_S)
        return UNVERIFIED
    except Exception:
        logger.warning("answer-key verification failed", exc_info=True)
        return UNVERIFIED
    return verdict


async def _verify(
    question: str,
    options: list[dict[str, str]],
    valid_labels: set[str],
    expected_label: str,
) -> str:
    client, model, binding = _build_client()
    messages = _messages(question, options)

    exec_tool = ExecTool()
    if can_use_native_tool_calling(binding=binding, model=model):
        text = await _run_tool_loop(
            client,
            model,
            binding,
            messages,
            exec_tool.get_definition().to_openai_schema(),
            exec_tool,
        )
    else:
        text = await _run_without_tools(client, model, binding, messages)

    verified = _parse_final(text, valid_labels)
    if not verified:
        logger.warning(
            "answer-key verification inconclusive: verifier reply had no usable "
            "FINAL line (last reply %.120r)",
            text,
        )
        return UNVERIFIED
    return AGREE if verified == expected_label.strip().upper() else DISAGREE


async def verify_short_answer(question: str, expected_answer: str) -> str:
    """Re-solve ``question`` independently and compare with ``expected_answer``.

    The same contract as :func:`verify_answer_key` for short questions: the
    solver never sees the registered answer, every calculation runs through
    the sandbox, and any infrastructure failure maps to ``"unverified"`` so
    the caller can fail open. The comparison is exact/numeric first; a
    phrasing-only difference falls through to the semantic-equivalence judge.
    """
    if not str(question or "").strip() or not str(expected_answer or "").strip():
        return UNVERIFIED
    if _network_calls_disabled():
        return UNVERIFIED
    try:
        return await asyncio.wait_for(
            _verify_short(question, expected_answer), timeout=TOTAL_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        logger.warning("short-answer verification timed out after %ss", TOTAL_TIMEOUT_S)
        return UNVERIFIED
    except Exception:
        logger.warning("short-answer verification failed", exc_info=True)
        return UNVERIFIED


async def _verify_short(question: str, expected_answer: str) -> str:
    solver_answer = await _solve_short(question)
    if not solver_answer:
        logger.warning("short-answer verification inconclusive: no usable FINAL line")
        return UNVERIFIED
    from deeptutor.learning.grading import grade_answer

    if grade_answer(solver_answer, expected_answer, "short"):
        return AGREE
    # The solver's answer differs from the registered key on surface form.
    # Before rejecting a registration, let the grader's own judge decide
    # whether the two are the same answer said differently.
    from deeptutor.capabilities.mastery.semantic_grade import (
        AGREE as JUDGE_AGREE,
    )
    from deeptutor.capabilities.mastery.semantic_grade import (
        DISAGREE as JUDGE_DISAGREE,
    )
    from deeptutor.capabilities.mastery.semantic_grade import (
        judge_free_text_answer,
    )

    verdict = await judge_free_text_answer(question, "short", expected_answer, solver_answer)
    if verdict == JUDGE_AGREE:
        return AGREE
    if verdict == JUDGE_DISAGREE:
        return DISAGREE
    return UNVERIFIED


async def _solve_short(question: str) -> str:
    client, model, binding = _build_client()
    messages = [
        {"role": "system", "content": _SHORT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Solve it (use exec for all arithmetic), then end with the FINAL line."
            ),
        },
    ]
    exec_tool = ExecTool()
    if can_use_native_tool_calling(binding=binding, model=model):
        text = await _run_tool_loop(
            client,
            model,
            binding,
            messages,
            exec_tool.get_definition().to_openai_schema(),
            exec_tool,
        )
    else:
        text = await _run_without_tools(client, model, binding, messages)
    return _parse_short_final(text)


def _parse_short_final(text: str) -> str:
    """Extract the solver's FINAL answer; return "" when unusable."""
    match = _SHORT_FINAL_RE.search(text or "")
    if not match:
        return ""
    answer = match.group(1).strip().strip("*$").strip()
    if not answer or answer.casefold() == "unsure":
        return ""
    return answer


def _build_client() -> tuple[Any, str | None, str | None]:
    """Resolve the task model (chat fallback) and build its OpenAI client."""
    from deeptutor.services.config import resolve_llm_runtime_config
    from deeptutor.services.model_selection.runtime import llm_config_from_resolved

    resolved = resolve_llm_runtime_config(service_name="task")
    if not (getattr(resolved, "model", "") or "").strip():
        # A task service with no model selected resolves to an empty model;
        # the verifier is worthless without one, so fall back to the chat
        # model rather than shipping the request to the provider unnamed.
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
    return (
        build_openai_client(client_config),
        client_config.model,
        client_config.binding,
    )


__all__ = [
    "AGREE",
    "DISAGREE",
    "UNVERIFIED",
    "verify_answer_key",
    "verify_short_answer",
]
