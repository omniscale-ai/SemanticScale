"""LLM self-verification of model-produced solutions for SH6.

Given a problem statement, the model's reasoning trace, and its extracted final
answer, ask the same (or any) LLM to judge whether the solution is correct.
The verifier never sees the reference answer — this is a self-check, comparable
to a self-consistency baseline but using a single attempt rather than a vote.

Output schema:

    {"verdict": "correct" | "incorrect",
     "confidence": int in [0, 100],
     "rationale": str}

The signed score ``+confidence`` for verdict="correct", ``-confidence`` for
verdict="incorrect" can be used as a best-of-N selection signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from semanticscale.llm_backend import Backend, make_backend

logger = logging.getLogger(__name__)


class SelfVerification(BaseModel):
    verdict: str = Field(description="Either 'correct' or 'incorrect'.")
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence in the verdict, 0 (unsure) to 100 (certain).",
    )
    rationale: str = Field(
        description="One- to three-sentence reasoning. Keep it short."
    )


_SYSTEM_PROMPT = (
    "You are an expert reviewer for science and mathematics problems.\n\n"
    "You will be shown:\n"
    "  1. A problem statement.\n"
    "  2. A student's reasoning trace.\n"
    "  3. The student's final answer.\n\n"
    "You do NOT have access to the reference answer. Judge the solution on its\n"
    "internal merits: is the reasoning sound, are the steps justified, and does\n"
    "the final answer follow from them?\n\n"
    "Return a JSON object with three fields:\n"
    "  - verdict: 'correct' if you think the answer is correct, 'incorrect' otherwise.\n"
    "  - confidence: integer 0-100. 100 = certain in your verdict, 0 = pure guess.\n"
    "  - rationale: 1-3 sentence justification. Be concise.\n\n"
    "Do not hedge: pick one verdict. Use 'confidence' to express your uncertainty."
)


def _user_prompt(problem: str, reasoning: str, final_answer: str) -> str:
    reasoning = (reasoning or "").strip()
    final_answer = (final_answer or "").strip()
    parts = [
        "## Problem",
        problem.strip(),
        "",
        "## Student reasoning",
        reasoning if reasoning else "(no reasoning trace provided)",
        "",
        "## Student final answer",
        final_answer if final_answer else "(no final answer provided)",
    ]
    return "\n".join(parts)


def signed_score(verdict: str, confidence: float | int) -> float:
    """Best-of-N selection signal: positive when correct, negative when incorrect."""
    sign = 1.0 if str(verdict).lower() == "correct" else -1.0
    return sign * float(confidence)


async def verify_one(
    backend: Backend,
    *,
    model: str,
    problem: str,
    reasoning: str,
    final_answer: str,
    semaphore: asyncio.Semaphore,
    service_tier: str | None = None,
    extra_body: dict | None = None,
) -> SelfVerification:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(problem, reasoning, final_answer)},
    ]
    async with semaphore:
        out = await backend.parse(
            model=model,
            messages=messages,
            text_format=SelfVerification,
            service_tier=service_tier,
            extra_body=extra_body,
        )
    verdict = "correct" if str(out.verdict).lower().startswith("correct") else "incorrect"
    return SelfVerification(
        verdict=verdict,
        confidence=int(max(0, min(100, out.confidence))),
        rationale=out.rationale,
    )


class VerifierCache:
    """Append-only JSONL cache keyed by attempt ``id``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: dict[str, dict] = {}

    def load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed cache line in %s", self.path)
                    continue
                key = str(rec.get("id"))
                if key:
                    self._entries[key] = rec

    def has(self, attempt_id: str) -> bool:
        return str(attempt_id) in self._entries

    def get(self, attempt_id: str) -> dict | None:
        return self._entries.get(str(attempt_id))

    def put(self, attempt_id: str, record: dict) -> None:
        self._entries[str(attempt_id)] = record

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in self._entries.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def entries(self) -> Iterable[dict]:
        return self._entries.values()


async def verify_attempts(
    attempts: list[dict],
    *,
    model_cfg: dict,
    cache: VerifierCache,
    max_concurrent: int = 20,
    flush_every: int = 25,
) -> list[dict]:
    """Run verifier over a list of attempt dicts.

    Each attempt dict must contain at least: ``id``, ``problem``,
    ``reasoning_text``, ``predicted_answer`` (or ``answer_text``), and
    ``is_correct``. Returns the same list with two added keys per attempt:
    ``verify_verdict``, ``verify_confidence`` (and ``verify_signed_score``).

    Verifier failures leave the attempt with ``verify_verdict=None``.
    """
    model = model_cfg["name"]
    service_tier = model_cfg.get("service_tier")
    extra_body = model_cfg.get("extra_body")

    backend = make_backend(model_cfg)
    semaphore = asyncio.Semaphore(max_concurrent)

    pending: list[dict] = []
    for att in attempts:
        if cache.has(att["id"]):
            continue
        pending.append(att)

    async def _run(att: dict) -> tuple[str, dict]:
        reasoning = att.get("reasoning_text") or ""
        final_answer = att.get("predicted_answer") or att.get("answer_text") or ""
        try:
            v = await verify_one(
                backend,
                model=model,
                problem=att["problem"],
                reasoning=reasoning,
                final_answer=final_answer,
                semaphore=semaphore,
                service_tier=service_tier,
                extra_body=extra_body,
            )
            rec = {
                "id": att["id"],
                "run": att.get("run"),
                "verdict": v.verdict,
                "confidence": int(v.confidence),
                "rationale": v.rationale,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Self-verification failed for id=%s: %s", att.get("id"), exc)
            rec = {
                "id": att["id"],
                "run": att.get("run"),
                "verdict": None,
                "confidence": None,
                "rationale": "",
                "error": str(exc),
            }
        return att["id"], rec

    try:
        completed = 0
        tasks = [_run(att) for att in pending]
        for coro in asyncio.as_completed(tasks):
            attempt_id, rec = await coro
            cache.put(attempt_id, rec)
            completed += 1
            if completed % flush_every == 0:
                cache.save()
                logger.info("Verifier progress: %d/%d", completed, len(tasks))
        cache.save()
    finally:
        await backend.aclose()

    enriched: list[dict] = []
    for att in attempts:
        out = dict(att)
        rec = cache.get(att["id"])
        if rec is None:
            out["verify_verdict"] = None
            out["verify_confidence"] = None
            out["verify_signed_score"] = None
            out["verify_rationale"] = ""
            out["verify_error"] = "no cache entry"
        else:
            out["verify_verdict"] = rec.get("verdict")
            out["verify_confidence"] = rec.get("confidence")
            out["verify_rationale"] = rec.get("rationale", "")
            out["verify_error"] = rec.get("error")
            if rec.get("verdict") is not None and rec.get("confidence") is not None:
                out["verify_signed_score"] = signed_score(rec["verdict"], rec["confidence"])
            else:
                out["verify_signed_score"] = None
        enriched.append(out)
    return enriched
