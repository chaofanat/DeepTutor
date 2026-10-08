"""mastery_quiz choice registrations are verified by an independent solver.

Three consecutive live mis-registered answer keys (each a constraint-
enumeration slip made while composing the question) graded correct learners
as incorrect until the tutor noticed and voided the attempt. The fix re-solves
every non-visual choice question before it is committed and rejects the
registration on a confident disagreement. These tests pin the contract:

* agreement registers, disagreement rejects without leaking the verifier's
  option, and any verification trouble fails open (the question registers);
* verification is never attempted for visual or non-choice questions;
* the FINAL-line parser behind the verdict accepts plain and decorated
  letters and rejects UNSURE / unknown labels / missing lines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeptutor.capabilities.mastery.tools import MasteryQuizTool
import deeptutor.capabilities.mastery.verify as verify_module
from deeptutor.capabilities.mastery.verify import AGREE, DISAGREE, UNVERIFIED
from deeptutor.learning.models import (
    KnowledgePoint,
    KnowledgeType,
    LearningModule,
    LearningProgress,
)
from deeptutor.learning.storage import LearningStore


def _use_store_root(monkeypatch, root: Path) -> None:
    def _init(self, root_arg=None):
        self._root = root / "learning"
        self._root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(LearningStore, "__init__", _init)


def _built_path(path_id: str = "path-1") -> LearningProgress:
    return LearningProgress(
        book_id=path_id,
        modules=[
            LearningModule(
                id="m1",
                name="Argebra",
                order=0,
                knowledge_points=[
                    KnowledgePoint(
                        id=f"{path_id}-kp1",
                        name="divisibility",
                        type=KnowledgeType.MEMORY,
                        module_id="m1",
                    )
                ],
            )
        ],
    )


def _patch_verdict(monkeypatch, verdict: str, calls: list | None = None) -> None:
    async def _verify(question, options, expected_label):
        if calls is not None:
            calls.append({"question": question, "options": options, "expected": expected_label})
        return verdict

    monkeypatch.setattr(verify_module, "verify_answer_key", _verify)


def _quiz_kwargs(**overrides) -> dict:
    kwargs = {
        "_mastery_path_id": "path-1",
        "knowledge_point_id": "path-1-kp1",
        "question": "How many n = 2^a*3^b satisfy the constraints?",
        "question_type": "choice",
        "options": [
            {"label": "A", "body": "2"},
            {"label": "B", "body": "3"},
            {"label": "C", "body": "4"},
            {"label": "D", "body": "6"},
        ],
        "expected_answer": "B",
        "explanation": "enumerate the pairs",
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.asyncio
async def test_agreement_registers_the_question(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    _patch_verdict(monkeypatch, AGREE)

    result = await MasteryQuizTool().execute(**_quiz_kwargs())

    assert result.success is True, result.content
    pending = LearningStore().load("path-1").pending_question
    assert pending is not None
    assert pending.question_type == "choice"


@pytest.mark.asyncio
async def test_disagreement_rejects_without_leaking_the_verifiers_option(
    tmp_path, monkeypatch
) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    _patch_verdict(monkeypatch, DISAGREE)

    result = await MasteryQuizTool().execute(**_quiz_kwargs())

    assert result.success is False
    # The rejection teaches the tutor that verification failed, but never
    # which option the verifier picked — the tool result is learner-visible.
    assert "disagreed" in result.content
    for option in _quiz_kwargs()["options"]:
        if option["label"] != "B":
            assert option["body"] not in result.content
    assert LearningStore().load("path-1").pending_question is None


@pytest.mark.asyncio
async def test_verification_trouble_fails_open(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    _patch_verdict(monkeypatch, UNVERIFIED)

    result = await MasteryQuizTool().execute(**_quiz_kwargs())

    assert result.success is True, result.content
    assert LearningStore().load("path-1").pending_question is not None


@pytest.mark.asyncio
async def test_visual_questions_skip_verification(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    calls: list[dict] = []
    _patch_verdict(monkeypatch, AGREE, calls)

    def _prepare_visual(*_args, **_kwargs):
        return {}, []

    monkeypatch.setattr("deeptutor.learning.visual_practice.prepare_visual", _prepare_visual)

    result = await MasteryQuizTool().execute(
        **_quiz_kwargs(
            visual={
                "task": "identification",
                "sources": [],
                "reference_quote": "quoted",
                "accepted_answers": [],
                "key_status": "unverified",
                "answer_cues": "none",
                "hints_used": 0,
            }
        )
    )

    # The visual branch owns its key handling; the independent verifier is
    # never consulted for it.
    assert calls == []
    assert result.success is True, result.content


@pytest.mark.asyncio
async def test_non_choice_questions_skip_verification(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    calls: list[dict] = []
    _patch_verdict(monkeypatch, AGREE, calls)

    result = await MasteryQuizTool().execute(
        **_quiz_kwargs(
            question_type="short",
            options=None,
            expected_answer="the derivative is 2x",
        )
    )

    assert calls == []
    assert result.success is True, result.content


def test_final_line_parser_acceptances() -> None:
    labels = {"A", "B", "C", "D"}
    assert verify_module._parse_final("work\nFINAL: C", labels) == "C"
    assert verify_module._parse_final("FINAL: **B**", labels) == "B"
    assert verify_module._parse_final("结论:FINAL：A", labels) == "A"
    assert verify_module._parse_final("FINAL: b", labels) == "B"


def test_final_line_parser_rejections() -> None:
    labels = {"A", "B"}
    assert verify_module._parse_final("FINAL: UNSURE", labels) == ""
    assert verify_module._parse_final("FINAL: E", labels) == ""
    assert verify_module._parse_final("never concluded", labels) == ""
    assert verify_module._parse_final("", labels) == ""
