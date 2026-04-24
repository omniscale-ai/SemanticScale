"""SWE-agent-trajectories dataset loader.

nebius/SWE-agent-trajectories ships ~80k SWE-agent runs that already
contain the agent's full reasoning trace, the final patch, and a
boolean ``target`` indicating whether the patch resolved the issue.
So Stage 1 here is "load and normalise" — no LLM inference is needed,
matching the ProcessBench pattern.

See https://huggingface.co/datasets/nebius/SWE-agent-trajectories.

Source row schema:
    instance_id      str        repo + issue identifier
    model_name       str        agent model that produced the trajectory
    target           bool       whether the patch solved the issue
    trajectory       list[dict] interleaved system/ai/user messages
    exit_status      str        agent termination reason
    generated_patch  str|None   final diff
    eval_logs        str|None   test logs

We treat each ``role == "ai"`` ``text`` field as one reasoning chunk.
The trajectory schema per entry is
``{cutoff_date, mask, role, system_prompt, text}`` — system entries
carry the prompt in ``system_prompt`` and a null ``text``; ai/user
entries carry their content in ``text``.
"""

from __future__ import annotations

import datetime
import logging
from datetime import timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_NAME = "swe-agent-trajectories"


def _models(config: dict, overrides: dict) -> list[str] | None:
    ds_cfg = config.get("dataset", {})
    models = overrides.get("models")
    if models is None:
        models = ds_cfg.get("models")
    if models in (None, "all"):
        return None
    return list(models)


def _targets(config: dict, overrides: dict) -> list[bool] | None:
    """Optional filter on the boolean ``target`` field.

    Accepts ``"all"``, ``None``, ``"success"``/``"fail"``, or a list of
    bools. Returns the bool values to keep, or None for "no filter".
    """
    ds_cfg = config.get("dataset", {})
    raw = overrides.get("targets")
    if raw is None:
        raw = ds_cfg.get("targets")
    if raw in (None, "all"):
        return None
    if isinstance(raw, str):
        raw = [raw]
    out: list[bool] = []
    for v in raw:
        if isinstance(v, bool):
            out.append(v)
        elif str(v).lower() in ("true", "success", "solved", "1"):
            out.append(True)
        elif str(v).lower() in ("false", "fail", "failed", "unsolved", "0"):
            out.append(False)
    return out or None


def _max_steps(config: dict, overrides: dict) -> int | None:
    ds_cfg = config.get("dataset", {})
    val = overrides.get("max_steps")
    if val is None:
        val = ds_cfg.get("max_steps")
    return int(val) if val is not None else None


def run_slug(config: dict, overrides: dict) -> str:
    """Identifier for a run: model filter + optional target/steps filters."""
    models = _models(config, overrides)
    targets = _targets(config, overrides)
    max_steps = _max_steps(config, overrides)

    parts: list[str] = []
    parts.append("model-" + "+".join(sorted(models)) if models else "model-all")
    if targets is not None:
        parts.append("target-" + "+".join("t" if t else "f" for t in sorted(targets)))
    if max_steps is not None:
        parts.append(f"steps-{max_steps}")
    return "_".join(parts)


def _extract_ai_chunks(trajectory: list, max_steps: int | None) -> list[str]:
    """Pull the ``text`` of every ``role == "ai"`` entry from the trajectory."""
    chunks: list[str] = []
    for entry in trajectory or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "ai":
            continue
        text = entry.get("text")
        if text is None:
            continue
        text = str(text).strip()
        if text:
            chunks.append(text)
    if max_steps is not None:
        chunks = chunks[:max_steps]
    return chunks


def produce_traces(
    config: dict,
    project_root: Path,
    overrides: dict,
) -> list[dict]:
    """Stage 1: load SWE-agent trajectories and normalise to trace records."""
    from datasets import load_dataset  # deferred: heavy dep

    ds_cfg = config.get("dataset", {})
    hf_path = ds_cfg.get("hf_path", "nebius/SWE-agent-trajectories")
    split = ds_cfg.get("split", "train")
    max_samples = overrides.get("max_samples")
    if max_samples is None:
        max_samples = ds_cfg.get("max_samples")

    model_filter = _models(config, overrides)
    target_filter = _targets(config, overrides)
    max_steps = _max_steps(config, overrides)

    slug = run_slug(config, overrides)
    ts = datetime.datetime.now(tz=timezone.utc).isoformat()

    logger.info("Loading %s split=%s", hf_path, split)
    ds = load_dataset(hf_path, split=split)

    traces: list[dict] = []
    skipped_no_chunks = 0
    for row in ds:
        if model_filter and row.get("model_name") not in model_filter:
            continue
        target = bool(row.get("target"))
        if target_filter is not None and target not in target_filter:
            continue

        chunks = _extract_ai_chunks(row.get("trajectory") or [], max_steps)
        if not chunks:
            skipped_no_chunks += 1
            continue

        instance_id = str(row.get("instance_id", ""))
        traces.append(
            {
                "id": instance_id,
                "dataset": DATASET_NAME,
                "run_slug": slug,
                # SWE-agent rows don't carry the original problem text, but
                # the instance_id encodes the repo + issue (e.g.
                # "django__django-12345") which is enough provenance.
                "problem": instance_id,
                "subject": str(row.get("model_name", "unknown")),
                "correct_answer": None,
                "reasoning_text": None,
                "answer_text": None,
                "reasoning_chunks": chunks,
                "answer_chunks": None,
                "is_correct": target,
                "error_step_index": None,
                "final_answer_correct": target,
                "generator": str(row.get("model_name", "")),
                "exit_status": str(row.get("exit_status", "")),
                "has_final_answer": bool(row.get("generated_patch")),
                "model": None,
                "usage": None,
                "error": None,
                "timestamp": ts,
                "grade": None,
            }
        )
        if max_samples is not None and len(traces) >= max_samples:
            break

    logger.info(
        "Loaded %d SWE-agent trajectories (models=%s, targets=%s, skipped_empty=%d)",
        len(traces),
        model_filter or "all",
        target_filter if target_filter is not None else "all",
        skipped_no_chunks,
    )
    return traces
