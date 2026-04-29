"""AgentHallu judgment and attribution evaluation for SH6.

Implements the benchmark's two prompting modes:

* ``all_at_once`` / standard prompting: judge the whole trajectory in one call,
  returning hallucination judgment, first responsible step, and explanation.
* ``step_by_step``: inspect prefixes incrementally and stop at the first step
  judged hallucinated; the current step becomes the attribution prediction.

The prompts and output schema follow the public AgentHallu reference code and
paper, but use SH6's backend abstraction so the evaluation runs with the same
OpenAI / OpenRouter / local-model config style as the rest of the experiment.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from semanticscale.llm_backend import Backend, make_backend

from . import scoring as generic_scoring

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant skilled in analyzing agent trajectory."

ALL_AT_ONCE_USER_PROMPT = """\
Given a question and an agent trajectory that contains all reasoning and action steps taken by the agent to answer the question. Your task is to determine whether the agent's final answer contains hallucinated content that contradicts the real-world facts or the question's ground truth. If hallucinated, you need to further identify the first step that introduced the hallucination, and explain the cause.

<Question>{problem}</Question>

<Agent Trajectory>{chat_content}</Agent Trajectory>

Based on the question and agent trajectory, please predict the following:
1. Examine the agent's answer in the given <Agent Trajectory> for the provided <Question>. If the answer contains hallucinated content, please set <is_hallucinated> to True, otherwise set <is_hallucinated> to False.
2. If <is_hallucinated> is True, identify the first step in the <Agent Trajectory> that introduces this hallucination and explain the cause. If <is_hallucinated> is False, set both hallucination_step and hallucination_reason to null.
"""

ALL_AT_ONCE_FORMAT_PROMPT = """
ABSOLUTELY do not return your reasoning process. Return ONLY one JSON object that matches the format below and nothing else (in particular, never start your response with "JSON" or "Result"):
{
    "is_hallucination": True/False: { "type": "boolean" },
    "hallucination_step": Step Number { "type": ["integer", "null"], "minimum": 1 },
    "hallucination_reason": Reason for Hallucination { "type": ["string", "null"], "maxLength": 1000},
}
"""

STEP_BY_STEP_USER_PROMPT = """\
Given a question and an agent trajectory up to the current reasoning and action steps taken by the agent to answer the question. Your task is to determine whether this most recent agent's step contains hallucinated content that could induce the question-solving process to produce an incorrect answer, and if so, explain the cause.

<Question>{problem}</Question>

<Agent Trajectory>{chat_content}</Agent Trajectory>

Based on the question and agent trajectory, please predict the following:
1. Examine the agent's answer in the given <Agent Trajectory> for the provided <Question>. If the answer contains hallucinated content, please set <is_hallucinated> to True, otherwise set <is_hallucinated> to False.
2. If <is_hallucinated> is True, explain the reason that introduces or causes this hallucination. If <is_hallucinated> is False, set hallucination_reason to null.
"""

STEP_BY_STEP_FORMAT_PROMPT = """
Return ONLY one JSON object that matches the format below and nothing else (in particular, never start your response with "JSON" or "Result"):
{
    "is_hallucination": True/False: { "type": "boolean" },
    "hallucination_reason": Reason for Hallucination { "type": ["string", "null"], "maxLength": 1000},
}
"""

GEVAL_PROMPT = """\
You will be given one evaluation instance consisting of a question, an hallucinated agent trajectory, an expected hallucination explanation, and a generated hallucination explanation. The agent trajectory is a hallucinated attempt to answer the question, where the expected explanation is a human-annotated description of the earliest decisive cause underlying the hallucination, and the generated explanation is the evaluator model's predicted attribution.

<Question>{problem}</Question>

<Agent Trajectory>{chat_content}</Agent Trajectory>

<Expected Explanation>{expected_explanation}</Expected Explanation>

<Generated Explanation>{generated_explanation}</Generated Explanation>

Your task is to evaluate the accuracy of generated explanation using the expected explanation as the gold reference.

Please make sure you read and understand following instructions carefully.

Evaluation Criteria:
Explanation Accuracy (1-5): the alignment between the generated explanation and the expected explanation in terms of error relevance, localization accuracy and causal correctness.
1. Score 1 (Fabricated): The explanation is distracted by irrelevant trajectory details and thus fails to attribute the error to the true hallucination cause, while also introducing fabricated evidence.
2. Score 2 (Mislocalized): The explanation references a genuine error in the trajectory but mislocalizes the hallucination by attributing it to a later step instead of the earliest decisive cause.
3. Score 3 (Wrong Cause at Correct Step): The explanation is grounded on the correct decisive step from the expected explanation, yet it misattributes the underlying cause of the hallucination.
4. Score 4 (Mostly Correct but Incomplete): The explanation matches the expected main cause with trajectory support, but is slightly incomplete or imprecise.
5. Score 5 (Exact Grounded): The explanation exactly matches the expected cause, is explicitly grounded in the trajectory, and adds no unsupported or contradictory content.

Evaluation Steps:
1. Read the question, agent trajectory, and expected explanation to establish the task intent and the trajectory segment where the hallucination becomes outcome-determining.
2. Identify the key attribution claims in the generated explanation including the claimed error source, the described error mechanism, and the specific trajectory step it relies on.
3. Compare the generated explanation to the expected explanation for the specified evaluation criterion in terms of error relevance, localization accuracy and causal correctness.
4. Assign a single explanation accuracy score in <1,2,3,4,5> according to the criteria.
5. Ignore formatting mismatches and evaluate based on content-level correspondence only.

Return ONLY one JSON object of the form {{"score": 1-5}}.
"""


class AllAtOncePrediction(BaseModel):
    is_hallucination: bool
    hallucination_step: int | None = Field(default=None, ge=1)
    hallucination_reason: str | None = Field(default=None, max_length=1000)


class StepByStepPrediction(BaseModel):
    is_hallucination: bool
    hallucination_reason: str | None = Field(default=None, max_length=1000)


class GevalScore(BaseModel):
    score: int = Field(ge=1, le=5)


def _eval_cfg(config: dict) -> dict:
    return config.get("agenthallu_eval") or {}


def enabled(config: dict) -> bool:
    cfg = _eval_cfg(config)
    return bool(cfg.get("enabled"))


def method(config: dict) -> str:
    cfg = _eval_cfg(config)
    return str(cfg.get("method") or "all_at_once")


def model_name(config: dict) -> str:
    cfg = _eval_cfg(config)
    return str(cfg.get("model", {}).get("name") or "")


def _geval_cfg(config: dict) -> dict:
    return _eval_cfg(config).get("geval") or {}


def geval_enabled(config: dict) -> bool:
    return bool(_geval_cfg(config).get("enabled"))


def _joined_trajectory(trace: dict, steps: list[dict] | None = None) -> str:
    payload = steps if steps is not None else trace.get("_agenthallu_eval_steps") or []
    return "\n".join(str(step["text"]) for step in payload)


def _prediction_fields(trace: dict) -> bool:
    return (
        "is_hallucination_prediction" in trace
        or "agenthallu_eval_error" in trace
        or "hallucination_geval_score" in trace
    )


def _ground_truth_is_hallucination(trace: dict) -> int:
    return 0 if trace.get("is_correct") else 1


def _ground_truth_step(trace: dict) -> int | None:
    raw = trace.get("hallucination_step")
    if raw is None:
        error_idx = trace.get("error_step_index")
        return None if error_idx is None else int(error_idx) + 1
    return int(raw)


def _prediction_answered(trace: dict) -> bool:
    return isinstance(trace.get("is_hallucination_prediction"), bool)


def _safe_prediction_step(trace: dict) -> int | None:
    raw = trace.get("hallucination_step_prediction")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _empty_prediction(trace: dict, *, error: str | None = None) -> dict:
    updated = dict(trace)
    updated["agenthallu_eval_method"] = updated.get("agenthallu_eval_method")
    updated["agenthallu_eval_model"] = updated.get("agenthallu_eval_model")
    updated["is_hallucination_prediction"] = None
    updated["hallucination_step_prediction"] = None
    updated["hallucination_reason_prediction"] = None
    updated["agenthallu_eval_error"] = error
    return updated


def _generated_explanation(trace: dict) -> str | None:
    step = _safe_prediction_step(trace)
    reason = trace.get("hallucination_reason_prediction")
    if step is None or not isinstance(reason, str) or not reason.strip():
        return None
    return f"The hallucination occurs at step {step}. {reason.strip()}"


def _empty_geval(trace: dict, *, score: float | None = None, error: str | None = None) -> dict:
    updated = dict(trace)
    updated["hallucination_geval_score"] = score
    updated["agenthallu_geval_error"] = error
    return updated


async def _parse_with_backend(
    backend: Backend,
    *,
    model: str,
    messages: list[dict],
    text_format: type[BaseModel],
    semaphore: asyncio.Semaphore,
    service_tier: str | None,
    extra_body: dict | None,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> BaseModel:
    async with semaphore:
        return await backend.parse(
            model=model,
            messages=messages,
            text_format=text_format,
            service_tier=service_tier,
            extra_body=extra_body,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
        )


def _all_at_once_messages(trace: dict) -> list[dict]:
    user_prompt = (
        ALL_AT_ONCE_USER_PROMPT.format(
            problem=trace["problem"],
            chat_content=_joined_trajectory(trace),
        )
        + "\n"
        + ALL_AT_ONCE_FORMAT_PROMPT
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _step_by_step_messages(problem: str, steps: list[dict]) -> list[dict]:
    user_prompt = (
        STEP_BY_STEP_USER_PROMPT.format(
            problem=problem,
            chat_content=_joined_trajectory({}, steps),
        )
        + "\n"
        + STEP_BY_STEP_FORMAT_PROMPT
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _geval_messages(trace: dict, generated_explanation: str) -> list[dict]:
    user_prompt = GEVAL_PROMPT.format(
        problem=trace["problem"],
        chat_content=_joined_trajectory(trace),
        expected_explanation=trace.get("hallucination_reason") or "",
        generated_explanation=generated_explanation,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


async def _evaluate_all_at_once(
    backend: Backend,
    trace: dict,
    *,
    model: str,
    semaphore: asyncio.Semaphore,
    service_tier: str | None,
    extra_body: dict | None,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> dict:
    try:
        pred: AllAtOncePrediction = await _parse_with_backend(
            backend,
            model=model,
            messages=_all_at_once_messages(trace),
            text_format=AllAtOncePrediction,
            semaphore=semaphore,
            service_tier=service_tier,
            extra_body=extra_body,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            retry_max_wait=retry_max_wait,
        )
    except Exception as exc:
        logger.warning("AgentHallu all-at-once evaluation failed for %s: %s", trace.get("id"), exc)
        return _empty_prediction(trace, error=str(exc))

    updated = dict(trace)
    updated["is_hallucination_prediction"] = pred.is_hallucination
    updated["hallucination_step_prediction"] = pred.hallucination_step if pred.is_hallucination else None
    updated["hallucination_reason_prediction"] = pred.hallucination_reason if pred.is_hallucination else None
    updated["agenthallu_eval_error"] = None
    return updated


async def _evaluate_step_by_step(
    backend: Backend,
    trace: dict,
    *,
    model: str,
    semaphore: asyncio.Semaphore,
    service_tier: str | None,
    extra_body: dict | None,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> dict:
    observed_steps: list[dict] = []
    for step in trace.get("_agenthallu_eval_steps") or []:
        observed_steps.append(step)
        try:
            pred: StepByStepPrediction = await _parse_with_backend(
                backend,
                model=model,
                messages=_step_by_step_messages(trace["problem"], observed_steps),
                text_format=StepByStepPrediction,
                semaphore=semaphore,
                service_tier=service_tier,
                extra_body=extra_body,
                max_retries=max_retries,
                retry_min_wait=retry_min_wait,
                retry_max_wait=retry_max_wait,
            )
        except Exception as exc:
            logger.warning("AgentHallu step-by-step evaluation failed for %s: %s", trace.get("id"), exc)
            return _empty_prediction(trace, error=str(exc))

        if pred.is_hallucination:
            updated = dict(trace)
            updated["is_hallucination_prediction"] = True
            updated["hallucination_step_prediction"] = int(step["step"])
            updated["hallucination_reason_prediction"] = pred.hallucination_reason
            updated["agenthallu_eval_error"] = None
            return updated

    updated = dict(trace)
    updated["is_hallucination_prediction"] = False
    updated["hallucination_step_prediction"] = None
    updated["hallucination_reason_prediction"] = None
    updated["agenthallu_eval_error"] = None
    return updated


async def _run_async(
    traces: list[dict],
    *,
    model_cfg: dict,
    model: str,
    prompt_method: str,
    max_concurrent: int,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> list[dict]:
    backend = make_backend(model_cfg)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _evaluate_indexed(idx: int, trace: dict) -> tuple[int, dict]:
        base = dict(trace)
        base["agenthallu_eval_method"] = prompt_method
        base["agenthallu_eval_model"] = model
        if prompt_method == "step_by_step":
            scored = await _evaluate_step_by_step(
                backend,
                base,
                model=model,
                semaphore=semaphore,
                service_tier=model_cfg.get("service_tier"),
                extra_body=model_cfg.get("extra_body"),
                max_retries=max_retries,
                retry_min_wait=retry_min_wait,
                retry_max_wait=retry_max_wait,
            )
        else:
            scored = await _evaluate_all_at_once(
                backend,
                base,
                model=model,
                semaphore=semaphore,
                service_tier=model_cfg.get("service_tier"),
                extra_body=model_cfg.get("extra_body"),
                max_retries=max_retries,
                retry_min_wait=retry_min_wait,
                retry_max_wait=retry_max_wait,
            )
        return idx, scored

    try:
        ordered: list[dict] = [{}] * len(traces)
        tasks = [_evaluate_indexed(i, trace) for i, trace in enumerate(traces)]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            idx, scored = await coro
            ordered[idx] = scored
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                logger.info("AgentHallu evaluation progress: %d/%d", completed, len(tasks))
        return ordered
    finally:
        await backend.aclose()


async def _run_geval_async(
    traces: list[dict],
    *,
    model_cfg: dict,
    model: str,
    max_concurrent: int,
    max_retries: int,
    retry_min_wait: float,
    retry_max_wait: float,
) -> list[dict]:
    backend = make_backend(model_cfg)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _evaluate_indexed(idx: int, trace: dict) -> tuple[int, dict]:
        if _ground_truth_is_hallucination(trace) != 1:
            return idx, _empty_geval(trace, score=None, error=None)

        generated_explanation = _generated_explanation(trace)
        if generated_explanation is None:
            return idx, _empty_geval(trace, score=0.0, error=None)

        try:
            pred: GevalScore = await _parse_with_backend(
                backend,
                model=model,
                messages=_geval_messages(trace, generated_explanation),
                text_format=GevalScore,
                semaphore=semaphore,
                service_tier=model_cfg.get("service_tier"),
                extra_body=model_cfg.get("extra_body"),
                max_retries=max_retries,
                retry_min_wait=retry_min_wait,
                retry_max_wait=retry_max_wait,
            )
        except Exception as exc:
            logger.warning("AgentHallu GEVAL failed for %s: %s", trace.get("id"), exc)
            return idx, _empty_geval(trace, score=None, error=str(exc))

        return idx, _empty_geval(trace, score=float(pred.score), error=None)

    try:
        ordered: list[dict] = [{}] * len(traces)
        tasks = [_evaluate_indexed(i, trace) for i, trace in enumerate(traces)]
        completed = 0
        for coro in asyncio.as_completed(tasks):
            idx, scored = await coro
            ordered[idx] = scored
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                logger.info("AgentHallu GEVAL progress: %d/%d", completed, len(tasks))
        return ordered
    finally:
        await backend.aclose()


def evaluate_traces(traces: list[dict], config: dict) -> list[dict]:
    """Run AgentHallu judgment/attribution evaluation when enabled."""
    if not enabled(config) or not traces:
        return traces

    cfg = _eval_cfg(config)
    model_cfg = dict(cfg.get("model") or {})
    model = str(model_cfg.get("name") or "")
    if not model:
        raise ValueError("agenthallu_eval.model.name is required when agenthallu_eval.enabled=true")

    prompt_method = method(config)
    if prompt_method not in {"all_at_once", "step_by_step"}:
        raise ValueError(
            f"Unsupported agenthallu_eval.method={prompt_method!r}. "
            "Use 'all_at_once' or 'step_by_step'."
        )

    scored = asyncio.run(
        _run_async(
            traces,
            model_cfg=model_cfg,
            model=model,
            prompt_method=prompt_method,
            max_concurrent=int(cfg.get("max_concurrent", 10)),
            max_retries=int(cfg.get("max_retries", 5)),
            retry_min_wait=float(cfg.get("retry_min_wait", 1.0)),
            retry_max_wait=float(cfg.get("retry_max_wait", 60.0)),
        )
    )
    if not geval_enabled(config):
        return scored

    geval_cfg = _geval_cfg(config)
    geval_model_cfg = dict(geval_cfg.get("model") or {})
    geval_model = str(geval_model_cfg.get("name") or "")
    if not geval_model:
        raise ValueError(
            "agenthallu_eval.geval.model.name is required when "
            "agenthallu_eval.geval.enabled=true"
        )
    return asyncio.run(
        _run_geval_async(
            scored,
            model_cfg=geval_model_cfg,
            model=geval_model,
            max_concurrent=int(geval_cfg.get("max_concurrent", 10)),
            max_retries=int(geval_cfg.get("max_retries", 5)),
            retry_min_wait=float(geval_cfg.get("retry_min_wait", 1.0)),
            retry_max_wait=float(geval_cfg.get("retry_max_wait", 60.0)),
        )
    )


def _metric_summary(rows: list[dict]) -> dict:
    total = len(rows)
    answered_rows = [row for row in rows if _prediction_answered(row)]
    errors = total - len(answered_rows)

    if answered_rows:
        y_true = [_ground_truth_is_hallucination(row) for row in answered_rows]
        y_pred = [1 if row["is_hallucination_prediction"] else 0 for row in answered_rows]
        correct = sum(int(gt == pred) for gt, pred in zip(y_true, y_pred))
        accuracy = accuracy_score(y_true, y_pred)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    else:
        correct = 0
        accuracy = 0.0
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    hallucinated_total = sum(_ground_truth_is_hallucination(row) for row in rows)
    hallucinated_answered_rows = [
        row for row in answered_rows if _ground_truth_is_hallucination(row) == 1
    ]
    step_correct = sum(
        int(_safe_prediction_step(row) == _ground_truth_step(row))
        for row in hallucinated_answered_rows
    )
    step_accuracy = (
        step_correct / len(hallucinated_answered_rows)
        if hallucinated_answered_rows
        else 0.0
    )
    geval_rows = [
        row for row in rows if _ground_truth_is_hallucination(row) == 1
    ]
    geval_scored_values = [
        float(row["hallucination_geval_score"])
        for row in geval_rows
        if row.get("hallucination_geval_score") is not None
    ]
    geval_score = (
        sum(geval_scored_values) / len(geval_scored_values)
        if geval_scored_values
        else 0.0
    )

    return {
        "total": total,
        "answered": len(answered_rows),
        "errors": errors,
        "correct": correct,
        "accuracy": float(accuracy),
        "judgment": {
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "accuracy": float(accuracy),
        },
        "attribution": {
            "step_localization_accuracy": float(step_accuracy),
            "geval_score": float(geval_score),
            "hallucinated_total": hallucinated_total,
            "hallucinated_answered": len(hallucinated_answered_rows),
            "step_correct": step_correct,
            "geval_scored": len(geval_scored_values),
        },
    }


def score_results(results: list[dict]) -> dict:
    """Compute AgentHallu benchmark metrics from SH6 trace records."""
    if results and not has_scored_predictions(results):
        return generic_scoring.score_results(results)
    if not results:
        return {
            "run_slug": "",
            "dataset": "agenthallu",
            "model": "",
            "evaluation_method": "",
            "total": 0,
            "answered": 0,
            "errors": 0,
            "correct": 0,
            "accuracy": 0.0,
            "by_subject": {},
            "judgment": {
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "accuracy": 0.0,
            },
            "attribution": {
                "step_localization_accuracy": 0.0,
                "geval_score": 0.0,
                "hallucinated_total": 0,
                "hallucinated_answered": 0,
                "step_correct": 0,
                "geval_scored": 0,
            },
            "by_hallucination_category": {},
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    first = results[0]
    summary = _metric_summary(results)
    by_subject_rows: dict[str, list[dict]] = defaultdict(list)
    by_category_rows: dict[str, list[dict]] = defaultdict(list)

    for row in results:
        by_subject_rows[str(row.get("subject") or "unknown")].append(row)
        category = row.get("hallucination_category") or "Unknown Type"
        by_category_rows[str(category)].append(row)

    return {
        "run_slug": first.get("run_slug", ""),
        "dataset": first.get("dataset", "agenthallu"),
        "model": first.get("agenthallu_eval_model", ""),
        "evaluation_method": first.get("agenthallu_eval_method", ""),
        "total": summary["total"],
        "answered": summary["answered"],
        "errors": summary["errors"],
        "correct": summary["correct"],
        "accuracy": summary["accuracy"],
        "by_subject": {
            label: {
                "total": slice_summary["total"],
                "correct": slice_summary["correct"],
                "errors": slice_summary["errors"],
                "accuracy": slice_summary["accuracy"],
            }
            for label, slice_summary in sorted(
                ((label, _metric_summary(rows)) for label, rows in by_subject_rows.items()),
                key=lambda item: item[0],
            )
        },
        "judgment": summary["judgment"],
        "attribution": summary["attribution"],
        "by_hallucination_category": {
            label: _metric_summary(rows)
            for label, rows in sorted(by_category_rows.items())
        },
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }


def has_scored_predictions(results: list[dict]) -> bool:
    return any(_prediction_fields(row) for row in results)
