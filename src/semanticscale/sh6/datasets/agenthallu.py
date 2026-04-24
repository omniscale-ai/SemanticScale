"""AgentHallu dataset loader.

liuxuannan/AgentHallu provides 693 hand-annotated multi-step agent trajectories
across 7 agent frameworks and 5 task domains, with a binary hallucination label,
first-hallucinated-step index, and hallucination category / reason annotations.

See https://github.com/liuxuannan/AgentHallu.

Per-sample fields (JSON files under ``AgentHallu/<framework>/NNN.json``):

    model_id                  str        backbone model (e.g. "Qwen_Qwen3-32B")
    agent_type                str        agent framework variant (e.g. "SmolAgents-ReAct")
    question                  str | list task prompt (nested list for BFCL multi-turn)
    true_answer               str | list ground-truth answer / tool-call sequence
    question_source           str        source benchmark tag
    question_domain           str        "Math" | "Science" | "Tool Use" |
                                         "World Knowledge" | "General Assistant"
    history                   list[dict] agent trajectory, step by step
    agent_answer              str        final natural-language response
    is_hallucination          str        "true" | "false"
    hallucination_step        str        1-indexed step where hallucination starts
                                         (present only when is_hallucination == "true")
    hallucination_category    str        coarse category
    hallucination_subcategory str        fine subtype
    hallucination_reason      str        human explanation

Each ``history[i]`` carries ``step``, optional ``role``, optional ``content``,
and optional ``tool_calls`` / ``tool_responses``. We flatten each step into a
single reasoning chunk by concatenating content with a compact rendering of
tool_calls. ``tool_responses`` are environment output (not agent reasoning) so
we drop them.
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
from datetime import timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_NAME = "agenthallu"

_DEFAULT_REPO_URL = "https://github.com/liuxuannan/AgentHallu.git"
_DEFAULT_REPO_REF = "main"

_ALL_FRAMEWORKS = (
    "BFCL",
    "Camel",
    "Magentic_One",
    "Octotools",
    "OpenDeepSearch",
    "OpenManus",
    "SmolAgents",
)


def _list_filter(config: dict, overrides: dict, key: str) -> list[str] | None:
    ds_cfg = config.get("dataset", {})
    val = overrides.get(key)
    if val is None:
        val = ds_cfg.get(key)
    if val in (None, "all"):
        return None
    if isinstance(val, str):
        val = [val]
    return list(val)


def _frameworks(config: dict, overrides: dict) -> list[str]:
    chosen = _list_filter(config, overrides, "frameworks")
    if chosen is None:
        return list(_ALL_FRAMEWORKS)
    unknown = [f for f in chosen if f not in _ALL_FRAMEWORKS]
    if unknown:
        raise ValueError(
            f"Unknown AgentHallu framework(s) {unknown}. "
            f"Known: {list(_ALL_FRAMEWORKS)}"
        )
    return chosen


def _domains(config: dict, overrides: dict) -> list[str] | None:
    return _list_filter(config, overrides, "domains")


def _models(config: dict, overrides: dict) -> list[str] | None:
    return _list_filter(config, overrides, "models")


def run_slug(config: dict, overrides: dict) -> str:
    """Identifier for a run: framework + optional domain/model filters."""
    frameworks = _frameworks(config, overrides)
    if set(frameworks) == set(_ALL_FRAMEWORKS):
        parts = ["framework-all"]
    else:
        parts = ["framework-" + "+".join(sorted(frameworks))]
    domains = _domains(config, overrides)
    if domains:
        parts.append(
            "domain-" + "+".join(sorted(d.replace(" ", "-") for d in domains))
        )
    models = _models(config, overrides)
    if models:
        parts.append("model-" + "+".join(sorted(models)))
    return "_".join(parts)


def _ensure_source(data_dir: Path, repo_url: str, ref: str) -> Path:
    """Clone AgentHallu into a cache dir on first use. Returns the data root."""
    cache_root = data_dir / "_sources" / f"agenthallu-{ref}"
    data_root = cache_root / "AgentHallu"
    if data_root.is_dir():
        return data_root
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning %s (ref=%s) into %s", repo_url, ref, cache_root)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ref,
            repo_url,
            str(cache_root),
        ],
        check=True,
    )
    if not data_root.is_dir():
        raise RuntimeError(
            f"Expected AgentHallu/ subdirectory inside {cache_root}, none found"
        )
    return data_root


def _stringify_tool_calls(tool_calls) -> str:
    if not tool_calls:
        return ""
    rendered: list[str] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            rendered.append(str(tc))
            continue
        name = tc.get("name", "tool")
        args = tc.get("arguments", "")
        if isinstance(args, (dict, list)):
            args = json.dumps(args, ensure_ascii=False)
        rendered.append(f"{name}({args})")
    return "Tool calls: " + " | ".join(rendered)


def _step_to_chunk(step: dict) -> str | None:
    """Flatten a history step into a single text chunk."""
    parts: list[str] = []
    content = step.get("content")
    if isinstance(content, str):
        text = content.strip()
        if text:
            parts.append(text)
    tool_calls = step.get("tool_calls")
    if tool_calls:
        rendered = _stringify_tool_calls(tool_calls)
        if rendered:
            parts.append(rendered)
    if not parts:
        return None
    return "\n".join(parts)


def _parse_step_index(raw) -> int | None:
    """Parse the dataset's 1-indexed ``hallucination_step`` into a 0-indexed int."""
    if raw is None:
        return None
    try:
        idx = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return max(0, idx - 1)


def _stringify(value) -> str:
    """AgentHallu questions/answers are sometimes nested lists (BFCL multi-turn)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def produce_traces(
    config: dict,
    project_root: Path,
    overrides: dict,
) -> list[dict]:
    """Stage 1: load AgentHallu trajectories and normalise to trace records."""
    ds_cfg = config.get("dataset", {})
    repo_url = ds_cfg.get("repo_url", _DEFAULT_REPO_URL)
    ref = ds_cfg.get("repo_ref", _DEFAULT_REPO_REF)
    local_path = ds_cfg.get("local_path")
    max_samples = overrides.get("max_samples")
    if max_samples is None:
        max_samples = ds_cfg.get("max_samples")

    framework_set = set(_frameworks(config, overrides))
    domain_filter = _domains(config, overrides)
    domain_set = set(domain_filter) if domain_filter else None
    model_filter = _models(config, overrides)
    model_set = set(model_filter) if model_filter else None

    data_dir = (project_root / config["paths"]["data_dir"]).resolve()

    if local_path:
        data_root = Path(local_path)
        if not data_root.is_absolute():
            data_root = (project_root / data_root).resolve()
        if not data_root.is_dir():
            raise FileNotFoundError(
                f"AgentHallu local_path {data_root} does not exist"
            )
        # The GitHub repo nests the data under an ``AgentHallu/`` subdirectory.
        # If the user points at the repo root, descend into it.
        if (data_root / "AgentHallu").is_dir() and not any(
            (data_root / f).is_dir() for f in _ALL_FRAMEWORKS
        ):
            data_root = data_root / "AgentHallu"
    else:
        data_root = _ensure_source(data_dir, repo_url, ref)

    slug = run_slug(config, overrides)
    ts = datetime.datetime.now(tz=timezone.utc).isoformat()

    traces: list[dict] = []
    skipped_no_chunks = 0
    for framework in sorted(framework_set):
        framework_dir = data_root / framework
        if not framework_dir.is_dir():
            logger.warning("Framework directory missing: %s", framework_dir)
            continue
        for json_path in sorted(framework_dir.glob("*.json")):
            with json_path.open("r", encoding="utf-8") as f:
                row = json.load(f)

            if domain_set and row.get("question_domain") not in domain_set:
                continue
            if model_set and row.get("model_id") not in model_set:
                continue

            history = row.get("history") or []
            chunks = [c for c in (_step_to_chunk(s) for s in history) if c]
            if not chunks:
                skipped_no_chunks += 1
                continue

            is_hallucination = str(
                row.get("is_hallucination", "")
            ).strip().lower() == "true"
            is_correct = not is_hallucination

            error_step = (
                _parse_step_index(row.get("hallucination_step"))
                if is_hallucination
                else None
            )
            if error_step is not None and error_step >= len(chunks):
                error_step = len(chunks) - 1

            agent_answer = row.get("agent_answer")
            answer_text = agent_answer if isinstance(agent_answer, str) else None

            traces.append(
                {
                    "id": f"{framework}/{json_path.stem}",
                    "dataset": DATASET_NAME,
                    "run_slug": slug,
                    "problem": _stringify(row.get("question")),
                    "subject": str(row.get("question_domain") or "unknown"),
                    "correct_answer": _stringify(row.get("true_answer")),
                    "reasoning_text": None,
                    "answer_text": answer_text,
                    "reasoning_chunks": chunks,
                    "answer_chunks": None,
                    "is_correct": is_correct,
                    "error_step_index": error_step,
                    "final_answer_correct": is_correct,
                    "generator": str(row.get("model_id") or ""),
                    "agent_type": str(row.get("agent_type") or ""),
                    "framework": framework,
                    "question_source": str(row.get("question_source") or ""),
                    "hallucination_category": row.get("hallucination_category"),
                    "hallucination_subcategory": row.get("hallucination_subcategory"),
                    "hallucination_reason": row.get("hallucination_reason"),
                    "model": None,
                    "usage": None,
                    "error": None,
                    "timestamp": ts,
                    "has_final_answer": agent_answer is not None,
                    "grade": None,
                }
            )
            if max_samples is not None and len(traces) >= max_samples:
                break
        if max_samples is not None and len(traces) >= max_samples:
            break

    logger.info(
        "Loaded %d AgentHallu trajectories "
        "(frameworks=%s, domains=%s, models=%s, skipped_empty=%d)",
        len(traces),
        sorted(framework_set),
        domain_filter or "all",
        model_filter or "all",
        skipped_no_chunks,
    )
    return traces
