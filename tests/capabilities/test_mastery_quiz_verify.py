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
async def test_short_questions_use_the_short_verifier_not_the_choice_one(
    tmp_path, monkeypatch
) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    choice_calls: list[dict] = []
    _patch_verdict(monkeypatch, AGREE, choice_calls)
    short_calls: list[str] = []

    async def _short_agree(question, expected_answer):
        short_calls.append(expected_answer)
        return AGREE

    monkeypatch.setattr(verify_module, "verify_short_answer", _short_agree)

    result = await MasteryQuizTool().execute(
        **_quiz_kwargs(
            question_type="short",
            options=None,
            expected_answer="the derivative is 2x",
        )
    )

    assert choice_calls == []
    assert short_calls == ["the derivative is 2x"]
    assert result.success is True, result.content


@pytest.mark.asyncio
async def test_short_answer_disagreement_rejects_registration(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())

    async def _short_disagree(_question, _expected_answer):
        return DISAGREE

    monkeypatch.setattr(verify_module, "verify_short_answer", _short_disagree)

    result = await MasteryQuizTool().execute(
        **_quiz_kwargs(
            question_type="short",
            options=None,
            expected_answer="6",
        )
    )

    assert result.success is False
    assert "disagreed" in result.content
    assert LearningStore().load("path-1").pending_question is None


@pytest.mark.asyncio
async def test_short_answer_trouble_fails_open(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())

    async def _short_unverified(_question, _expected_answer):
        return UNVERIFIED

    monkeypatch.setattr(verify_module, "verify_short_answer", _short_unverified)

    result = await MasteryQuizTool().execute(
        **_quiz_kwargs(
            question_type="short",
            options=None,
            expected_answer="8",
        )
    )

    assert result.success is True, result.content
    assert LearningStore().load("path-1").pending_question is not None


@pytest.mark.asyncio
async def test_short_verification_compares_before_judging(monkeypatch) -> None:
    # Exact and numeric equivalence settle the comparison without the judge;
    # only a phrasing difference escalates to it.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    judge_calls: list[dict] = []

    async def _solve(question):
        return "1/2"

    async def _judge(question, question_type, expected_answer, learner_answer):
        judge_calls.append({"expected": expected_answer, "learner": learner_answer})
        return "agree"

    monkeypatch.setattr(verify_module, "_solve_short", _solve)
    monkeypatch.setattr(
        "deeptutor.capabilities.mastery.semantic_grade.judge_free_text_answer", _judge
    )

    verdict = await verify_module.verify_short_answer("What is half of one?", "0.5")

    assert verdict == AGREE
    assert judge_calls == []


@pytest.mark.asyncio
async def test_short_verification_escalates_phrasing_differences_to_the_judge(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    async def _solve(question):
        return "在叶绿体中"

    async def _judge(question, question_type, expected_answer, learner_answer):
        assert (expected_answer, learner_answer) == ("叶绿体", "在叶绿体中")
        return "agree"

    monkeypatch.setattr(verify_module, "_solve_short", _solve)
    monkeypatch.setattr(
        "deeptutor.capabilities.mastery.semantic_grade.judge_free_text_answer", _judge
    )

    verdict = await verify_module.verify_short_answer("光合作用发生在哪里？", "叶绿体")

    assert verdict == AGREE


@pytest.mark.asyncio
async def test_short_verification_judge_disagreement_rejects(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    async def _solve(question):
        return "8"

    async def _judge(_question, _question_type, _expected_answer, _learner_answer):
        return "disagree"

    monkeypatch.setattr(verify_module, "_solve_short", _solve)
    monkeypatch.setattr(
        "deeptutor.capabilities.mastery.semantic_grade.judge_free_text_answer", _judge
    )

    verdict = await verify_module.verify_short_answer("divisors of 1800 divisible by 5 not 3?", "6")

    assert verdict == DISAGREE


@pytest.mark.asyncio
async def test_short_verification_unusable_solver_reply_fails_open(monkeypatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    async def _solve(question):
        return ""

    monkeypatch.setattr(verify_module, "_solve_short", _solve)

    verdict = await verify_module.verify_short_answer("q", "8")

    assert verdict == UNVERIFIED


def test_short_final_parser_acceptances() -> None:
    assert verify_module._parse_short_final("work\nFINAL: 8") == "8"
    assert verify_module._parse_short_final("FINAL: **$8$**") == "8"
    assert verify_module._parse_short_final("FINAL：叶绿体") == "叶绿体"
    assert verify_module._parse_short_final("FINAL: 0.5") == "0.5"


def test_short_final_parser_rejections() -> None:
    assert verify_module._parse_short_final("FINAL: UNSURE") == ""
    assert verify_module._parse_short_final("never concluded") == ""
    assert verify_module._parse_short_final("") == ""


@pytest.mark.asyncio
async def test_the_verifier_is_inert_under_pytest() -> None:
    # The kill-switch itself: unpatched, inside pytest, verification fails
    # open without spending a real provider call — the suite (and CI) has
    # always assumed this, including on machines with saved credentials.
    verdict = await verify_module.verify_answer_key(
        "2+2=?", [{"label": "A", "body": "3"}, {"label": "B", "body": "4"}], "B"
    )

    assert verdict == UNVERIFIED


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
