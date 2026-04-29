"""AgentErrorBench dataset loader.

Supports local or mirrored snapshots of AgentErrorBench trajectories for SH6.
Stage 1 normalises trajectory files into the common SH6 trace schema; no LLM
inference is required.

Supported input shapes:

1. AgentDebug-style trajectory JSON with ``metadata`` plus ``messages`` or
   legacy ``chat_history``.
2. Episode JSON/JSONL records produced by
   ``agentdebug.rollout.step_to_episode`` (``messages`` / ``steps`` /
   ``final_response`` / ``won``).
3. Step-level JSONL rollouts with ``prompt`` / ``action`` rows; these are
   grouped into episodes automatically.
4. Optional detector outputs (for example ``*_critical_error.json``) that carry
   ``critical_error`` annotations. When matching trajectories are present, SH6
   uses the annotated critical step as ``error_step_index``.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import subprocess
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATASET_NAME = "agenterrorbench"
SLICE_NAME = "environment"

_DEFAULT_FILE_GLOBS = ("**/*.jsonl", "**/*.json")
_CRITICAL_STEP_RE = re.compile(r"Step\s+(\d+)", re.IGNORECASE)
_DEFAULT_DRIVE_URL = (
    "https://drive.google.com/drive/folders/"
    "1bQe6dQA85pktT63YnKIKJDTVaH3O3Vpu?usp=drive_link"
)


def slice_label(trace: dict) -> str | None:
    """Return the benchmark environment for per-environment analysis grouping."""
    env = trace.get("environment")
    return str(env) if env else None


def _string_list_filter(config: dict, overrides: dict, key: str) -> list[str] | None:
    ds_cfg = config.get("dataset", {})
    value = overrides.get(key)
    if value is None:
        value = ds_cfg.get(key)
    if value in (None, "all"):
        return None
    if isinstance(value, str):
        value = [value]
    return [str(item) for item in value]


def _environments(config: dict, overrides: dict) -> list[str] | None:
    return _string_list_filter(config, overrides, "environments")


def _providers(config: dict, overrides: dict) -> list[str] | None:
    return _string_list_filter(config, overrides, "providers")


def _models(config: dict, overrides: dict) -> list[str] | None:
    return _string_list_filter(config, overrides, "models")


def _slug_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", value).strip("-") or "unknown"


def run_slug(config: dict, overrides: dict) -> str:
    """Identifier for a run: environment + optional provider/model filters."""
    environments = _environments(config, overrides)
    providers = _providers(config, overrides)
    models = _models(config, overrides)
    parts = [
        "env-" + "+".join(sorted(_slug_token(v) for v in environments))
        if environments
        else "env-all",
        "provider-" + "+".join(sorted(_slug_token(v) for v in providers))
        if providers
        else "provider-all",
        "model-" + "+".join(sorted(_slug_token(v) for v in models))
        if models
        else "model-all",
    ]
    return "_".join(parts)


def _resolve_source_root(config: dict, project_root: Path) -> Path:
    ds_cfg = config.get("dataset", {})
    local_path = ds_cfg.get("local_path")
    if local_path:
        root = Path(local_path)
        if not root.is_absolute():
            root = (project_root / root).resolve()
        if not root.exists():
            drive_url = ds_cfg.get("drive_url") or _DEFAULT_DRIVE_URL
            if ds_cfg.get("download_if_missing", True):
                _download_drive_folder(root, drive_url)
            else:
                raise FileNotFoundError(
                    f"AgentErrorBench local_path {root} does not exist"
                )
        return root

    repo_url = ds_cfg.get("repo_url")
    if not repo_url:
        raise FileNotFoundError(
            "AgentErrorBench requires dataset.local_path or dataset.repo_url. "
            "Point local_path at a directory or file containing trajectory JSON/JSONL."
        )

    ref = str(ds_cfg.get("repo_ref") or "main")
    data_dir = (project_root / config["paths"]["data_dir"]).resolve()
    cache_root = data_dir / "_sources" / f"agenterrorbench-{ref}"
    if not cache_root.exists():
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

    source_subdir = ds_cfg.get("source_subdir")
    root = cache_root / source_subdir if source_subdir else cache_root
    if not root.exists():
        raise FileNotFoundError(
            f"AgentErrorBench source_subdir {root} does not exist inside cloned repo"
        )
    return root


def _download_drive_folder(output_root: Path, drive_url: str) -> None:
    """Download the public AgentErrorBench Drive folder into *output_root*.

    Some individual Drive files can intermittently reject programmatic access.
    Download everything else, log the skipped paths, and only fail when nothing
    usable lands on disk.
    """
    try:
        import gdown
        from gdown.exceptions import FileURLRetrievalError
    except ImportError as exc:
        raise ImportError(
            "AgentErrorBench auto-download requires gdown. "
            "Install project dependencies with `uv sync`."
        ) from exc

    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading AgentErrorBench from Google Drive into %s", output_root)
    planned = gdown.download_folder(
        url=drive_url,
        output=str(output_root),
        quiet=True,
        skip_download=True,
    )
    if not planned:
        raise RuntimeError(
            f"AgentErrorBench Google Drive listing returned no files for {drive_url}"
        )

    skipped: list[str] = []
    available = 0
    for item in planned:
        local_path = Path(item.local_path)
        if local_path.exists() and local_path.stat().st_size > 0:
            available += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = gdown.download(
                id=item.id,
                output=str(local_path),
                quiet=True,
                resume=True,
            )
        except FileURLRetrievalError as exc:
            skipped.append(item.path)
            logger.warning("Skipping AgentErrorBench file %s: %s", item.path, exc)
            continue

        if result:
            available += 1

    if available == 0:
        raise RuntimeError(
            "AgentErrorBench download failed; no files were retrieved from Google Drive."
        )

    if skipped:
        logger.warning(
            "Skipped %d AgentErrorBench files during download: %s",
            len(skipped),
            ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""),
        )

    logger.info(
        "AgentErrorBench download complete: %d/%d files available under %s",
        available,
        len(planned),
        output_root,
    )


def _iter_candidate_files(root: Path, config: dict) -> list[Path]:
    if root.is_file():
        return [root]
    ds_cfg = config.get("dataset", {})
    patterns = ds_cfg.get("file_globs") or list(_DEFAULT_FILE_GLOBS)
    files: set[Path] = set()
    for pattern in patterns:
        files.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(files)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                records.append(row)
    return records


def _coerce_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("records", "episodes", "trajectories", "data", "items"):
            nested = payload.get(key)
            if isinstance(nested, list) and all(isinstance(item, dict) for item in nested):
                return list(nested)
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "1", "yes", "y", "success", "passed", "won"}:
            return True
        if normalized in {"false", "f", "0", "no", "n", "fail", "failed", "lost"}:
            return False
    return None


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    return _stringify(content).strip()


def _trajectory_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if isinstance(messages, list) and all(isinstance(item, dict) for item in messages):
        return list(messages)
    chat_history = record.get("chat_history")
    if isinstance(chat_history, list) and all(isinstance(item, dict) for item in chat_history):
        return list(chat_history)

    messages = []
    for step in record.get("steps") or []:
        if not isinstance(step, dict):
            continue
        current_input = _stringify(step.get("current_input")).strip()
        if current_input:
            messages.append({"role": "user", "content": current_input})
        content = _stringify(step.get("content") or step.get("action")).strip()
        if content:
            messages.append({"role": "assistant", "content": content})
        env_response = _stringify(step.get("env_response") or step.get("observation")).strip()
        if env_response:
            messages.append({"role": "user", "content": env_response})
    return messages


def _extract_reasoning_chunks(record: dict[str, Any]) -> list[str]:
    messages = _trajectory_messages(record)
    chunks = [
        _message_content(message)
        for message in messages
        if str(message.get("role", "")).strip().lower() == "assistant"
        and _message_content(message)
    ]
    if chunks:
        return chunks

    reason_text = _stringify(record.get("reason_text")).strip()
    return [reason_text] if reason_text else []


def _extract_problem(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    for candidate in (
        record.get("task_description"),
        metadata.get("task"),
        metadata.get("task_description"),
        metadata.get("prompt"),
        record.get("prompt"),
    ):
        text = _stringify(candidate).strip()
        if text:
            return text

    for message in _trajectory_messages(record):
        if str(message.get("role", "")).strip().lower() == "user":
            text = _message_content(message)
            if text:
                return text
    return ""


def _extract_answer_text(record: dict[str, Any]) -> str | None:
    for candidate in (
        record.get("answer_text"),
        record.get("final_response"),
        record.get("normalized_response"),
    ):
        text = _stringify(candidate).strip()
        if text:
            return text

    assistant_messages = [
        _message_content(message)
        for message in _trajectory_messages(record)
        if str(message.get("role", "")).strip().lower() == "assistant"
        and _message_content(message)
    ]
    return assistant_messages[-1] if assistant_messages else None


def _extract_correct_answer(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata") or {}
    for candidate in (
        record.get("ground_truth"),
        metadata.get("ground_truth"),
        record.get("correct_answer"),
        metadata.get("correct_answer"),
        record.get("answer"),
    ):
        text = _stringify(candidate).strip()
        if text:
            return text
    return None


def _looks_like_step_row(record: dict[str, Any]) -> bool:
    return (
        "prompt" in record
        and "action" in record
        and "messages" not in record
        and "chat_history" not in record
    )


def _step_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("environment"),
        row.get("batch_idx"),
        row.get("test_idx"),
        row.get("attempt_idx", 0),
        row.get("env_id"),
    )


def _step_rows_to_episode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered_rows = sorted(rows, key=lambda row: row.get("step", 0))
    first = ordered_rows[0]
    last = ordered_rows[-1]

    messages: list[dict[str, str]] = []
    steps: list[dict[str, Any]] = []
    task_description = ""
    for idx, row in enumerate(ordered_rows, start=1):
        prompt = _stringify(row.get("prompt")).strip()
        action = _stringify(row.get("action")).strip()
        if prompt:
            messages.append({"role": "user", "content": prompt})
            if not task_description:
                task_description = prompt
        if action:
            messages.append({"role": "assistant", "content": action})
        steps.append(
            {
                "step": idx,
                "current_input": prompt,
                "content": action,
                "env_response": "",
            }
        )

    metadata = {
        "task_id": first.get("task_id")
        or first.get("episode_id")
        or f"{first.get('environment', 'env')}-{first.get('env_id', 'unknown')}",
        "task": task_description,
        "environment": first.get("environment"),
        "success": last.get("success", last.get("won")),
        "won": last.get("won"),
        "provider": first.get("provider"),
        "model": first.get("model"),
    }
    return {
        "metadata": metadata,
        "messages": messages,
        "steps": steps,
        "episode_id": first.get("episode_id"),
        "task_key": first.get("task_key"),
        "provider": first.get("provider"),
        "model": first.get("model"),
        "environment": first.get("environment"),
        "task_success": last.get("success", last.get("won")),
        "won": last.get("won"),
        "final_response": _stringify(last.get("action")).strip(),
    }


def _annotation_key(record: dict[str, Any], path: Path) -> list[str]:
    keys: list[str] = []
    metadata = record.get("metadata") or {}
    for candidate in (
        record.get("task_id"),
        record.get("episode_id"),
        record.get("id"),
        record.get("task_key"),
        metadata.get("task_id"),
        metadata.get("id"),
    ):
        text = _stringify(candidate).strip()
        if text:
            keys.append(text)
    stem = path.stem
    keys.append(stem)
    if stem.endswith("_critical_error"):
        keys.append(stem.removesuffix("_critical_error"))
    return keys


def _looks_like_label_record(record: dict[str, Any]) -> bool:
    return "trajectory_id" in record and "critical_failure_step" in record


def _critical_error_from_label(record: dict[str, Any]) -> dict[str, Any]:
    module = _stringify(record.get("critical_failure_module")).strip() or None
    error_type = None
    reasoning = None

    for step_annotation in record.get("step_annotations") or []:
        if not isinstance(step_annotation, dict):
            continue
        payload = step_annotation.get(module) if module else None
        if not isinstance(payload, dict):
            for key, value in step_annotation.items():
                if key != "step" and isinstance(value, dict):
                    module = key
                    payload = value
                    break
        if isinstance(payload, dict):
            error_type = payload.get("failure_type")
            reasoning = payload.get("reasoning")
            break

    return {
        "critical_step": record.get("critical_failure_step"),
        "critical_module": module,
        "error_type": error_type,
        "root_cause": reasoning,
        "confidence": None,
    }


def _looks_like_annotation_only(record: dict[str, Any]) -> bool:
    return (
        "critical_error" in record
        and not record.get("messages")
        and not record.get("chat_history")
        and not record.get("steps")
    )


def _looks_like_trajectory(record: dict[str, Any]) -> bool:
    return any(
        key in record
        for key in (
            "messages",
            "chat_history",
            "steps",
            "reason_text",
            "final_response",
            "normalized_response",
            "metadata",
        )
    )


def _parse_error_step_index(record: dict[str, Any]) -> int | None:
    critical_error = record.get("critical_error")
    if isinstance(critical_error, dict):
        critical_step = critical_error.get("critical_step")
        if critical_step is not None:
            try:
                return max(0, int(critical_step) - 1)
            except (TypeError, ValueError):
                pass

    error_summary = record.get("error_summary")
    if isinstance(error_summary, dict):
        critical_at = _stringify(error_summary.get("critical_at"))
        match = _CRITICAL_STEP_RE.search(critical_at)
        if match:
            return max(0, int(match.group(1)) - 1)

    explicit = record.get("error_step_index")
    if explicit is not None:
        try:
            return max(0, int(explicit))
        except (TypeError, ValueError):
            return None
    return None


def _extract_outcome(record: dict[str, Any]) -> bool | None:
    metadata = record.get("metadata") or {}
    for candidate in (
        record.get("final_answer_correct"),
        record.get("is_correct"),
        record.get("task_success"),
        record.get("success"),
        record.get("won"),
        metadata.get("final_answer_correct"),
        metadata.get("is_correct"),
        metadata.get("success"),
        metadata.get("won"),
    ):
        parsed = _as_bool(candidate)
        if parsed is not None:
            return parsed

    if record.get("critical_error") is not None:
        return False
    return None


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        text = _stringify(value).strip()
        if text:
            return text
    return None


def produce_traces(
    config: dict,
    project_root: Path,
    overrides: dict,
) -> list[dict]:
    """Stage 1: load AgentErrorBench trajectories and normalise to trace records."""
    ds_cfg = config.get("dataset", {})
    max_samples = overrides.get("max_samples")
    if max_samples is None:
        max_samples = ds_cfg.get("max_samples")

    environment_filter = _environments(config, overrides)
    provider_filter = _providers(config, overrides)
    model_filter = _models(config, overrides)
    environment_set = set(environment_filter) if environment_filter else None
    provider_set = set(provider_filter) if provider_filter else None
    model_set = set(model_filter) if model_filter else None

    source_root = _resolve_source_root(config, project_root)
    files = _iter_candidate_files(source_root, config)
    if not files and ds_cfg.get("download_if_missing", True):
        drive_url = ds_cfg.get("drive_url") or _DEFAULT_DRIVE_URL
        if source_root.is_dir():
            _download_drive_folder(source_root, drive_url)
            files = _iter_candidate_files(source_root, config)
    logger.info("Scanning %d AgentErrorBench files under %s", len(files), source_root)
    if not files:
        raise FileNotFoundError(
            f"No AgentErrorBench JSON/JSONL files found under {source_root}"
        )

    annotation_index: dict[str, dict[str, Any]] = {}
    label_index: dict[str, dict[str, Any]] = {}
    trajectories: list[tuple[dict[str, Any], Path]] = []

    for path in files:
        try:
            payload = _load_jsonl(path) if path.suffix == ".jsonl" else _load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.debug("Skipping unreadable file %s: %s", path, exc)
            continue

        records = _coerce_records(payload)
        if records and all(_looks_like_step_row(record) for record in records):
            grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in records:
                grouped_rows[_step_group_key(row)].append(row)
            for rows in grouped_rows.values():
                trajectories.append((_step_rows_to_episode(rows), path))
            continue

        for record in records:
            if _looks_like_label_record(record):
                label_id = _stringify(record.get("trajectory_id")).strip()
                if label_id:
                    label_index[label_id] = record
                continue
            if _looks_like_annotation_only(record):
                for key in _annotation_key(record, path):
                    annotation_index[key] = record
                continue
            if _looks_like_trajectory(record):
                trajectories.append((record, path))

    slug = run_slug(config, overrides)
    timestamp = datetime.datetime.now(tz=timezone.utc).isoformat()

    traces: list[dict] = []
    skipped_missing_outcome = 0
    skipped_empty = 0

    for index, (record, path) in enumerate(trajectories):
        metadata = record.get("metadata") or {}
        raw_id = _first_nonempty(
            record.get("task_id"),
            record.get("episode_id"),
            record.get("id"),
            metadata.get("task_id"),
            metadata.get("id"),
            record.get("task_key"),
        )
        if raw_id is None:
            raw_id = path.stem

        label = label_index.get(raw_id) or label_index.get(path.stem)
        annotation = None
        for key in (raw_id, path.stem, path.stem.removesuffix("_critical_error")):
            if key in annotation_index:
                annotation = annotation_index[key]
                break
        if label and "critical_error" not in record:
            merged = dict(record)
            merged["critical_error"] = _critical_error_from_label(label)
            merged["task_success"] = False
            merged["is_correct"] = False
            merged["final_answer_correct"] = False
            if "environment" not in merged:
                merged["environment"] = label.get("task_type")
            if "model" not in merged:
                merged["model"] = label.get("LLM")
            record = merged
            metadata = record.get("metadata") or {}
        if annotation and "critical_error" not in record:
            merged = dict(record)
            merged["critical_error"] = annotation.get("critical_error")
            merged["error_summary"] = annotation.get("error_summary")
            if "task_success" not in merged and "task_success" in annotation:
                merged["task_success"] = annotation.get("task_success")
            record = merged
            metadata = record.get("metadata") or {}

        environment = _first_nonempty(
            record.get("environment"),
            metadata.get("environment"),
            label.get("task_type") if label else None,
            path.parent.name,
            record.get("subject"),
        ) or "unknown"
        provider = _first_nonempty(record.get("provider"), metadata.get("provider"))
        generator = _first_nonempty(
            record.get("model"),
            metadata.get("model"),
            label.get("LLM") if label else None,
        )

        if environment_set and environment not in environment_set:
            continue
        if provider_set and provider not in provider_set:
            continue
        if model_set and generator not in model_set:
            continue

        is_correct = _extract_outcome(record)
        if is_correct is None:
            skipped_missing_outcome += 1
            continue

        reasoning_chunks = _extract_reasoning_chunks(record)
        if not reasoning_chunks:
            skipped_empty += 1
            continue

        answer_text = _extract_answer_text(record)
        error_step_index = _parse_error_step_index(record)
        if error_step_index is not None and error_step_index >= len(reasoning_chunks):
            error_step_index = len(reasoning_chunks) - 1

        critical_error = record.get("critical_error")
        critical_module = critical_type = critical_root_cause = None
        critical_confidence = None
        if isinstance(critical_error, dict):
            critical_module = critical_error.get("critical_module")
            critical_type = critical_error.get("error_type")
            critical_root_cause = critical_error.get("root_cause")
            critical_confidence = critical_error.get("confidence")

        traces.append(
            {
                "id": raw_id,
                "dataset": DATASET_NAME,
                "run_slug": slug,
                "problem": _extract_problem(record),
                "subject": environment,
                "correct_answer": _extract_correct_answer(record),
                "reasoning_text": None,
                "answer_text": answer_text,
                "reasoning_chunks": reasoning_chunks,
                "answer_chunks": None,
                "is_correct": is_correct,
                "error_step_index": error_step_index,
                "final_answer_correct": is_correct,
                "environment": environment,
                "provider": provider,
                "generator": generator,
                "task_key": _first_nonempty(record.get("task_key")),
                "critical_error_module": critical_module,
                "critical_error_type": critical_type,
                "critical_error_confidence": critical_confidence,
                "critical_error_root_cause": critical_root_cause,
                "model": None,
                "usage": None,
                "error": None,
                "timestamp": timestamp,
                "has_final_answer": bool(answer_text),
                "grade": None,
            }
        )
        if max_samples is not None and len(traces) >= max_samples:
            break

    logger.info(
        "Loaded %d AgentErrorBench trajectories "
        "(envs=%s, providers=%s, models=%s, skipped_missing_outcome=%d, skipped_empty=%d)",
        len(traces),
        environment_filter or "all",
        provider_filter or "all",
        model_filter or "all",
        skipped_missing_outcome,
        skipped_empty,
    )
    return traces
