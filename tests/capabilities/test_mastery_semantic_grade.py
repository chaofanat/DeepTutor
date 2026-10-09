"""Free-text mastery answers get a semantic-equivalence second chance.

The deterministic grader compares surface forms. A learner who answers
「两地相距三百千米」 against the reference 「距离为300公里」 is correct and
was graded wrong every time — and a wrong grade feeds the error records and
the spaced-repetition scheduler, locking the objective into endless review.
These tests pin the two-layer contract:

* the judge is invoked only for non-blank short/open answers the
  deterministic matcher already ruled wrong, and never for choice questions,
  already-graded replays, or blank submissions;
* an AGREE verdict rescues the answer while the learner's real wording stays
  in the attempt record; DISAGREE / UNVERIFIED / judge crashes keep the
  deterministic verdict;
* the FINAL-line parser behind the verdict accepts plain and decorated forms
  and rejects missing or unknown verdicts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import deeptutor.capabilities.mastery.semantic_grade as semantic_grade_module
from deeptutor.capabilities.mastery.semantic_grade import AGREE, DISAGREE, UNVERIFIED
from deeptutor.capabilities.mastery.tools import MasteryGradeTool, MasteryQuizTool
import deeptutor.capabilities.mastery.verify as verify_module
from deeptutor.learning.models import (
    ErrorType,
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
                name="Distance",
                order=0,
                knowledge_points=[
                    KnowledgePoint(
                        id=f"{path_id}-kp1",
                        name="distance phrasing",
                        type=KnowledgeType.MEMORY,
                        module_id="m1",
                    )
                ],
            )
        ],
    )


def _patch_judge(monkeypatch, verdict: str | None, calls: list | None = None) -> None:
    async def _judge(question, question_type, expected_answer, learner_answer):
        if calls is not None:
            calls.append(
                {
                    "question": question,
                    "question_type": question_type,
                    "expected": expected_answer,
                    "learner": learner_answer,
                }
            )
        if verdict is None:  # None simulates an infrastructure crash.
            raise RuntimeError("judge exploded")
        return verdict

    monkeypatch.setattr(semantic_grade_module, "judge_free_text_answer", _judge)


async def _register_short_question(
    *, expected: str = "距离为300公里", question: str = "两地相距多远？"
) -> str:
    result = await MasteryQuizTool().execute(
        _mastery_path_id="path-1",
        knowledge_point_id="path-1-kp1",
        question=question,
        question_type="short",
        expected_answer=expected,
        explanation="read it off the map.",
    )
    assert result.success is True, result.content
    pending = LearningStore().load("path-1").pending_question
    assert pending is not None
    return pending.question_id


def _payload(result) -> dict:
    assert result.success is True, result.content
    return json.loads(result.content)


@pytest.mark.asyncio
async def test_agree_rescues_a_paraphrased_answer(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    question_id = await _register_short_question()
    _patch_judge(monkeypatch, AGREE)

    result = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer="两地相距三百千米"
    )

    payload = _payload(result)
    assert payload["is_correct"] is True
    assert payload["semantic_equivalent"] is True
    progress = LearningStore().load("path-1")
    attempt = next(a for a in progress.quiz_attempts if a.question_id == question_id)
    assert attempt.is_correct is True
    # The learner's real wording is the record — the semantic verdict must
    # not rewrite history into the reference answer.
    assert attempt.user_answer == "两地相距三百千米"
    interaction = LearningStore().get_interaction("path-1", question_id)
    assert interaction.result["semantic_equivalent"] is True


@pytest.mark.asyncio
async def test_disagree_keeps_the_deterministic_verdict(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    question_id = await _register_short_question()
    _patch_judge(monkeypatch, DISAGREE)

    result = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer="两地相距五百千米"
    )

    payload = _payload(result)
    assert payload["is_correct"] is False
    assert "semantic_equivalent" not in payload
    progress = LearningStore().load("path-1")
    attempt = next(a for a in progress.quiz_attempts if a.question_id == question_id)
    assert attempt.is_correct is False
    assert attempt.error_type is ErrorType.APPLICATION_ERROR
    assert len(progress.error_records) == 1


@pytest.mark.asyncio
async def test_unverified_keeps_the_deterministic_verdict(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    question_id = await _register_short_question()
    _patch_judge(monkeypatch, UNVERIFIED)

    result = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer="两地相距五百千米"
    )

    assert _payload(result)["is_correct"] is False


@pytest.mark.asyncio
async def test_judge_internal_crash_maps_to_unverified(monkeypatch) -> None:
    # judge_free_text_answer owns its failure handling: any internal crash
    # must surface as UNVERIFIED, never propagate into the grade transaction.
    # Drop the pytest network kill-switch so this reaches the real guard.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(semantic_grade_module, "_judge", _boom)

    verdict = await semantic_grade_module.judge_free_text_answer("q?", "short", "ref", "ans")

    assert verdict == semantic_grade_module.UNVERIFIED


@pytest.mark.asyncio
async def test_deterministically_correct_answers_skip_the_judge(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    question_id = await _register_short_question()
    calls: list[dict] = []
    _patch_judge(monkeypatch, DISAGREE, calls)

    result = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer="答案是距离为300公里"
    )

    assert _payload(result)["is_correct"] is True
    assert calls == []


@pytest.mark.asyncio
async def test_choice_answers_never_invoke_the_judge(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())

    async def _agree(_question, _options, _expected):
        return verify_module.AGREE

    monkeypatch.setattr(verify_module, "verify_answer_key", _agree)
    quiz = await MasteryQuizTool().execute(
        _mastery_path_id="path-1",
        knowledge_point_id="path-1-kp1",
        question="Which notation equals one half?",
        question_type="choice",
        options=[
            {"label": "A", "body": "0.5"},
            {"label": "B", "body": "0.6"},
            {"label": "C", "body": "0.05"},
        ],
        expected_answer="A",
        explanation="half.",
    )
    assert quiz.success is True, quiz.content
    pending = LearningStore().load("path-1").pending_question
    question_id = pending.question_id
    # The shuffle already moved the labels around; pick a label whose body is
    # not the registered key so the graded answer is genuinely wrong.
    wrong_label = next(
        option.label for option in pending.options if option.label != pending.expected_answer
    )
    calls: list[dict] = []
    _patch_judge(monkeypatch, AGREE, calls)

    result = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer=wrong_label
    )

    assert _payload(result)["is_correct"] is False
    assert calls == []


@pytest.mark.asyncio
async def test_blank_answers_never_invoke_the_judge(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    question_id = await _register_short_question()
    calls: list[dict] = []
    _patch_judge(monkeypatch, AGREE, calls)

    result = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer="   "
    )

    payload = _payload(result)
    assert payload["is_correct"] is False
    assert calls == []
    progress = LearningStore().load("path-1")
    attempt = next(a for a in progress.quiz_attempts if a.question_id == question_id)
    assert attempt.error_type is ErrorType.METACOGNITIVE


@pytest.mark.asyncio
async def test_already_graded_replay_does_not_reinvoke_the_judge(tmp_path, monkeypatch) -> None:
    _use_store_root(monkeypatch, tmp_path)
    LearningStore().save(_built_path())
    question_id = await _register_short_question()
    calls: list[dict] = []
    _patch_judge(monkeypatch, AGREE, calls)

    first = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer="两地相距三百千米"
    )
    assert _payload(first)["is_correct"] is True
    assert len(calls) == 1

    second = await MasteryGradeTool().execute(
        _mastery_path_id="path-1", question_id=question_id, answer="两地相距三百千米"
    )
    payload = _payload(second)
    assert payload["is_correct"] is True
    assert payload["replayed"] is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_judge_is_inert_under_pytest() -> None:
    # The kill-switch itself: unpatched, inside pytest, the judge must fail
    # closed instead of spending a real provider call.
    verdict = await semantic_grade_module.judge_free_text_answer("q?", "short", "ref", "ans")

    assert verdict == semantic_grade_module.UNVERIFIED


def test_verdict_parser_acceptances() -> None:
    assert semantic_grade_module._parse_verdict("reasoning\nFINAL: AGREE") == "agree"
    assert semantic_grade_module._parse_verdict("FINAL: **DISAGREE**") == "disagree"
    assert semantic_grade_module._parse_verdict("结论:FINAL：UNSURE") == "unsure"


def test_verdict_parser_rejections() -> None:
    assert semantic_grade_module._parse_verdict("FINAL: MAYBE") == ""
    assert semantic_grade_module._parse_verdict("never concluded") == ""
    assert semantic_grade_module._parse_verdict("") == ""
