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
import logging
import os
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
    api_type: str,
    service_tier: str | None = None,
    extra_body: dict | None = None,
) -> object:
    """Call parse() with retry, returning the parsed Pydantic object."""

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
            
        if extra_body:
            kwargs["extra_body"] = extra_body
            
        if api_type == "completions":
            response = await client.beta.chat.completions.parse(
                model=grader_model,
                messages=messages,
                response_format=text_format,
                **kwargs,
            )
            return response.choices[0].message.parsed
        else:
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
    api_type: str,
    service_tier: str | None = None,
    extra_body: dict | None = None,
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
            client, messages, FinalAnswerGrade, grader_model, semaphore, api_type, service_tier, extra_body
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
    api_type: str,
    service_tier: str | None = None,
    extra_body: dict | None = None,
) -> dict:
    """Grade an open-ended problem using a rubric; returns a grade dict."""
    rubric_items = _parse_rubric_items(result["correct_answer"])
    model_answer = result.get("answer_text", "(no answer provided)")

    if rubric_items:
        # One LLM call per rubric criterion.
        async def _grade_one(rubric_item: dict) -> GradeRubricItem:
            max_pts = rubric_item["max_point_per_item"]
            item_desc = rubric_item["item_description"]
            system_prompt = (
                "You are an expert grader for science and mathematics problems.\n\n"
                "Problem statement:\n"
                f"{result['problem']}\n\n"
                f"Grading criterion (max {max_pts} points):\n"
                f"{item_desc}\n\n"
                "Grade the student's response on this single criterion only. "
                f"awarded_points must be between 0 and {max_pts}."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": model_answer},
            ]
            return await _parse_response(
                client, messages, GradeRubricItem, grader_model, semaphore, api_type, service_tier, extra_body
            )

        raw = await asyncio.gather(
            *[_grade_one(ri) for ri in rubric_items], return_exceptions=True
        )

        items = []
        for i, (rubric_item, graded) in enumerate(zip(rubric_items, raw)):
            ref_max = float(rubric_item["max_point_per_item"])
            if isinstance(graded, Exception):
                logger.warning(
                    "Grading failed for item %s criterion %d: %s",
                    result.get("id"), i, graded,
                )
                items.append({
                    "awarded_points": 0.0,
                    "explanation": f"Grading error: {graded}",
                    "max_points": ref_max,
                    "error": str(graded),
                })
                continue
            if float(graded.awarded_points) > ref_max:
                logger.warning(
                    "RubricGrade awarded_points %.1f exceeds max_points %.1f "
                    "for item %s, criterion %d; clamping",
                    float(graded.awarded_points), ref_max, result.get("id"), i,
                )
            awarded = max(0.0, min(float(graded.awarded_points), ref_max))
            items.append({
                "awarded_points": awarded,
                "explanation": graded.explanation,
                "max_points": ref_max,
            })

        total_max = sum(it["max_points"] for it in items)
        total_awarded = sum(it["awarded_points"] for it in items)
        if total_max != 10.0:
            logger.warning(
                "Rubric total_max=%.1f for item %s (expected 10)",
                total_max, result.get("id"),
            )
        return {
            "type": "rubric",
            "items": items,
            "total_max": total_max,
            "total_awarded": total_awarded,
        }

    # No structured rubric items — single call with the raw correct_answer.
    system_prompt = (
        "You are an expert grader for science and mathematics problems.\n\n"
        "Problem statement:\n"
        f"{result['problem']}\n\n"
        "Grading rubric:\n"
        f"{result['correct_answer']}\n\n"
        "Grade the student's response according to the rubric. "
        "Return a list of rubric items with one entry per rubric criterion. "
        "For each item specify: "
        "awarded_points (the points awarded) and explanation (why those points were awarded)."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": model_answer},
    ]
    try:
        rubric: RubricGrade = await _parse_response(
            client, messages, RubricGrade, grader_model, semaphore, api_type, service_tier, extra_body
        )
        items = [
            {"awarded_points": max(0.0, float(item.awarded_points)), "explanation": item.explanation}
            for item in rubric.items
        ]
        total_awarded = sum(it["awarded_points"] for it in items)
        return {
            "type": "rubric",
            "items": items,
            "total_max": 0.0,
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
    grading_config = config.get("grading", {})
    model_config = config.get("model", {})
    max_concurrent = config.get("inference", {}).get("max_concurrent", 10)
    service_tier = grading_config.get("service_tier")
    
    api_type = grading_config.get("api_type", model_config.get("api_type", "responses"))
    base_url = grading_config.get("base_url", model_config.get("base_url"))
    api_key_env = grading_config.get("api_key_env", model_config.get("api_key_env"))
    extra_body = grading_config.get("extra_body", model_config.get("extra_body"))
    
    return asyncio.run(_run_async(results, grader_model, max_concurrent, service_tier, api_type, base_url, api_key_env, extra_body))


async def _run_async(
    results: list[dict],
    grader_model: str,
    max_concurrent: int,
    service_tier: str | None = None,
    api_type: str = "responses",
    base_url: str | None = None,
    api_key_env: str | None = None,
    extra_body: dict | None = None,
) -> list[dict]:
    api_key = os.environ.get(api_key_env) if api_key_env else os.environ.get("OPENAI_API_KEY")
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _grade_indexed(idx: int, result: dict) -> tuple[int, dict]:
        if result.get("error"):
            return idx, {}
        if result.get("has_final_answer", True):
            grade = await _grade_final_answer(client, result, grader_model, semaphore, api_type, service_tier, extra_body)
        else:
            grade = await _grade_rubric(client, result, grader_model, semaphore, api_type, service_tier, extra_body)
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
