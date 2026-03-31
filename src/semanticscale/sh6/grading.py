"""LLM-based advanced grading for SH6 results.

Two grading modes are supported:

* FINAL ANSWER problems — the grader receives the problem, the reference answer,
  and the model's extracted final answer.  It returns a pass/fail decision with
  a textual explanation.

* Open-ended (no FINAL ANSWER) problems — the grader receives the problem, the
  reference answer / grading rubric, and the full model response.  It returns a
  list of GradeRubricItem(max_points, awarded_points, explanation).
"""

import asyncio
import json
import logging
import re

import openai
import tenacity
from pydantic import BaseModel, Field

from semanticscale.openai_utils import should_retry_openai_exception

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------


class FinalAnswerGrade(BaseModel):
    passed: bool
    explanation: str


class GradeRubricItem(BaseModel):
    awarded_points: float = Field(ge=0)
    explanation: str


class RubricGrade(BaseModel):
    items: list[GradeRubricItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RUBRIC_ITEM_RE = re.compile(r"Points:\s*([\d.]+)\s*,\s*Item:\s*", re.IGNORECASE)


def _parse_rubric_items(correct_answer: str) -> list[dict]:
    """Split correct_answer into structured rubric items.

    Each item starts with 'Points: <float>, Item:' and extends to the next
    such marker or end of string.  Returns a list of
    ``{"max_point_per_item": float, "item_description": str}`` dicts.
    Returns an empty list when no markers are found.
    """
    matches = list(_RUBRIC_ITEM_RE.finditer(correct_answer))
    if not matches:
        return []
    items = []
    for i, match in enumerate(matches):
        points = float(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(correct_answer)
        description = correct_answer[start:end].strip()
        items.append({"max_point_per_item": points, "item_description": description})
    return items


_FINAL_ANSWER_RE = re.compile(
    r"FINAL ANSWER\b\s*[:\-]?\s*(.*?)(?=\n[A-Z][A-Z ]{2,}\b|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_after_final_answer(answer_text: str) -> str:
    """Return the raw text that follows the 'FINAL ANSWER' marker."""
    matches = list(_FINAL_ANSWER_RE.finditer(answer_text))
    if matches:
        return matches[-1].group(1).strip()
    return answer_text.strip()


# ---------------------------------------------------------------------------
# Per-item grading coroutines
# ---------------------------------------------------------------------------


async def _parse_response(
    client: openai.AsyncOpenAI,
    messages: list[dict],
    text_format: type,
    grader_model: str,
    semaphore: asyncio.Semaphore,
    service_tier: str | None = None,
) -> object:
    """Call responses.parse() with retry, returning the parsed Pydantic object."""

    @tenacity.retry(
        retry=tenacity.retry_if_exception(should_retry_openai_exception),
        wait=tenacity.wait_exponential(min=1.0, max=60.0),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def _call() -> object:
        kwargs = {}
        if service_tier is not None:
            kwargs["service_tier"] = service_tier
        response = await client.responses.parse(
            model=grader_model,
            input=messages,
            text_format=text_format,
            **kwargs,
        )
        return response.output_parsed

    async with semaphore:
        return await _call()


async def _grade_final_answer(
    client: openai.AsyncOpenAI,
    result: dict,
    grader_model: str,
    semaphore: asyncio.Semaphore,
    service_tier: str | None = None,
) -> dict:
    """Grade a FINAL ANSWER problem; returns a grade dict."""
    system_prompt = (
        "You are an expert grader for science and mathematics problems.\n\n"
        "Problem statement:\n"
        f"{result['problem']}\n\n"
        "Reference answer:\n"
        f"{result['correct_answer']}\n\n"
        "The student was instructed to write their answer after \"FINAL ANSWER\".\n"
        "You will receive only the portion of their response that comes after that marker.\n"
        "Determine whether the student's answer is mathematically and scientifically "
        "equivalent to the reference answer. Minor differences in notation or formatting "
        "that are mathematically equivalent should be marked as passed."
    )
    model_answer = _extract_after_final_answer(result.get("answer_text", ""))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": model_answer or "(no answer provided)"},
    ]
    try:
        grade: FinalAnswerGrade = await _parse_response(
            client, messages, FinalAnswerGrade, grader_model, semaphore, service_tier
        )
        return {
            "type": "final_answer",
            "passed": grade.passed,
            "explanation": grade.explanation,
        }
    except Exception as exc:
        logger.warning("Grading failed for item %s: %s", result.get("id"), exc)
        return {
            "type": "final_answer",
            "passed": None,
            "explanation": f"Grading error: {exc}",
            "error": str(exc),
        }


async def _grade_rubric(
    client: openai.AsyncOpenAI,
    result: dict,
    grader_model: str,
    semaphore: asyncio.Semaphore,
    service_tier: str | None = None,
) -> dict:
    """Grade an open-ended problem using a rubric; returns a grade dict."""
    rubric_items = _parse_rubric_items(result["correct_answer"])
    if rubric_items:
        rubric_content = json.dumps(rubric_items, ensure_ascii=False)
        rubric_instruction = (
            "The rubric is provided as a JSON array. "
            "Each element has max_point_per_item (maximum total points for that criterion) "
            "and item_description (what must be demonstrated to earn those points). "
            "Item description may include multiple additive point awards - they must add up to max_point_per_item."
        )
    else:
        rubric_content = result["correct_answer"]
        rubric_instruction = ""
    system_prompt = (
        "You are an expert grader for science and mathematics problems.\n\n"
        "Problem statement:\n"
        f"{result['problem']}\n\n"
        "Grading rubric:\n"
        f"{rubric_content}\n\n"
        f"{rubric_instruction}"
        "Grade the student's response according to the rubric. "
        "Return a list of rubric items with one entry per rubric criterion. "
        "For each item specify: "
        "max_points (the maximum points for that criterion), "
        "awarded_points (between 0 and max_points), "
        "and explanation (why those points were awarded)."
    )
    model_answer = result.get("answer_text", "(no answer provided)")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": model_answer},
    ]
    try:
        rubric: RubricGrade = await _parse_response(
            client, messages, RubricGrade, grader_model, semaphore, service_tier
        )
        if rubric_items and len(rubric.items) != len(rubric_items):
            logger.warning(
                "RubricGrade item count mismatch for item %s: "
                "grader returned %d items, rubric has %d",
                result.get("id"),
                len(rubric.items),
                len(rubric_items),
            )
        items = []
        for i, item in enumerate(rubric.items):
            ref_max = (
                float(rubric_items[i]["max_point_per_item"])
                if rubric_items and i < len(rubric_items)
                else None
            )
            if ref_max is not None and float(item.awarded_points) > ref_max:
                logger.warning(
                    "RubricGrade awarded_points %.1f exceeds max_points %.1f "
                    "for item %s, criterion %d; clamping",
                    float(item.awarded_points),
                    ref_max,
                    result.get("id"),
                    i,
                )
            awarded = (
                max(0.0, min(float(item.awarded_points), ref_max))
                if ref_max is not None
                else max(0.0, float(item.awarded_points))
            )
            entry = {"awarded_points": awarded, "explanation": item.explanation}
            if ref_max is not None:
                entry["max_points"] = ref_max
            items.append(entry)
        total_max = sum(it["max_points"] for it in items if "max_points" in it)
        total_awarded = sum(it["awarded_points"] for it in items)
        if total_max != 10.0:
            logger.warning(
                "Rubric total_max=%.1f for item %s (expected 10)",
                total_max,
                result.get("id"),
            )
        return {
            "type": "rubric",
            "items": items,
            "total_max": total_max,
            "total_awarded": total_awarded,
        }
    except Exception as exc:
        logger.warning("Grading failed for item %s: %s", result.get("id"), exc)
        return {
            "type": "rubric",
            "items": [],
            "total_max": 0.0,
            "total_awarded": 0.0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grade_results(results: list[dict], grader_model: str, config: dict) -> list[dict]:
    """Call the grader model on each result and attach a ``grade`` field.

    For FINAL ANSWER problems the grade is::

        {"type": "final_answer", "passed": bool, "explanation": str}

    For open-ended (rubric) problems::

        {"type": "rubric",
         "items": [{"max_points": float, "awarded_points": float, "explanation": str}, ...],
         "total_max": float, "total_awarded": float}

    ``is_correct`` is updated to reflect the advanced grade:
    - FINAL ANSWER: ``grade["passed"]``
    - Rubric: ``total_awarded >= 7.0`` (expects total_max == 10; warns if not)

    Returns a new list of result dicts with ``grade`` and updated ``is_correct``.
    """
    max_concurrent = config.get("inference", {}).get("max_concurrent", 10)
    service_tier = config.get("grading", {}).get("service_tier")
    return asyncio.run(_run_async(results, grader_model, max_concurrent, service_tier))


async def _run_async(
    results: list[dict],
    grader_model: str,
    max_concurrent: int,
    service_tier: str | None = None,
) -> list[dict]:
    client = openai.AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _grade_indexed(idx: int, result: dict) -> tuple[int, dict]:
        if result.get("error"):
            return idx, {}
        if result.get("has_final_answer", True):
            grade = await _grade_final_answer(client, result, grader_model, semaphore, service_tier)
        else:
            grade = await _grade_rubric(client, result, grader_model, semaphore, service_tier)
        return idx, grade

    tasks = [_grade_indexed(i, r) for i, r in enumerate(results)]
    ordered_grades: list[dict] = [{}] * len(results)
    completed = 0
    for coro in asyncio.as_completed(tasks):
        idx, grade = await coro
        ordered_grades[idx] = grade
        completed += 1
        if completed % 50 == 0 or completed == len(tasks):
            logger.info("Grading progress: %d/%d", completed, len(tasks))

    updated = []
    for result, grade in zip(results, ordered_grades):
        r = dict(result)
        if grade:
            r["grade"] = grade
            if "error" not in grade:
                if grade["type"] == "final_answer" and grade["passed"] is not None:
                    r["is_correct"] = grade["passed"]
                elif grade["type"] == "rubric":
                    total_max = grade.get("total_max", 0.0)
                    total_awarded = grade.get("total_awarded", 0.0)
                    r["is_correct"] = total_awarded >= 7.0 if total_max > 0 else False
        updated.append(r)
    return updated
