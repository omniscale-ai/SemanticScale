"""SLoD pairwise ranking for SH6 reasoning trajectories.

Chunks texts by \\n\\n, runs double LLM comparisons (a→b, b→a) for
consensus, caches results to disk, and uses choix to compute
Bradley-Terry parameters (higher = more abstract).
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

import choix
import numpy as np
import openai
import tenacity
from pydantic import BaseModel

from semanticscale.openai_utils import should_retry_openai_exception

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are evaluating the Semantic Level of Detail (SLoD) of scientific text passages.\n\n"
    "Higher abstraction (macro level): general principles, concepts, strategy, high-level overviews.\n"
    "Lower abstraction (micro level): specific calculations, formulas, numbers, concrete steps.\n\n"
    "Compare the two passages below and decide which is at a HIGHER level of abstraction."
)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------


class AbstractionWinner(BaseModel):
    winner: Literal["A", "B"]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str) -> list[str]:
    """Split text by \\n\\n and return non-empty stripped chunks."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]


# ---------------------------------------------------------------------------
# Comparison cache
# ---------------------------------------------------------------------------


class ComparisonCache:
    """JSON-backed directional cache mapping 'hash_a:hash_b' → 'A' | 'B'.

    'hash_a:hash_b' → 'A' means chunk_a was presented as Passage A,
    chunk_b as Passage B, and A won (chunk_a is more abstract).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = {}

    def load(self) -> None:
        if self._path.exists():
            with self._path.open(encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info("Loaded %d cached comparisons from %s", len(self._data), self._path)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._data, f)
        tmp.replace(self._path)
        logger.debug("Saved %d comparisons to cache", len(self._data))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    def _key(self, chunk_a: str, chunk_b: str) -> str:
        return f"{self._hash(chunk_a)}:{self._hash(chunk_b)}"

    def get(self, chunk_a: str, chunk_b: str) -> str | None:
        """Return 'A' or 'B' if cached (A = chunk_a is more abstract), else None."""
        return self._data.get(self._key(chunk_a, chunk_b))

    def set(self, chunk_a: str, chunk_b: str, result: str) -> None:
        """Store result 'A' or 'B' (A = chunk_a is more abstract)."""
        self._data[self._key(chunk_a, chunk_b)] = result


# ---------------------------------------------------------------------------
# Single LLM comparison
# ---------------------------------------------------------------------------


async def _compare_once(
    client: openai.AsyncOpenAI,
    model: str,
    service_tier: str | None,
    chunk_a: str,
    chunk_b: str,
    cache: ComparisonCache,
    semaphore: asyncio.Semaphore,
) -> str:
    """Ask the LLM which chunk is more abstract (A or B).

    chunk_a is presented as Passage A, chunk_b as Passage B.
    Returns 'A' if chunk_a is more abstract, 'B' if chunk_b is.
    """
    cached = cache.get(chunk_a, chunk_b)
    if cached is not None:
        return cached

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Passage A:\n{chunk_a}\n\n"
                f"Passage B:\n{chunk_b}"
            ),
        },
    ]

    @tenacity.retry(
        retry=tenacity.retry_if_exception(should_retry_openai_exception),
        wait=tenacity.wait_exponential(min=1.0, max=60.0),
        stop=tenacity.stop_after_attempt(5),
        reraise=True,
    )
    async def _call() -> str:
        kwargs: dict = {}
        if service_tier is not None:
            kwargs["service_tier"] = service_tier
        response = await client.responses.parse(
            model=model,
            input=messages,
            text_format=AbstractionWinner,
            **kwargs,
        )
        return response.output_parsed.winner

    async with semaphore:
        result = await _call()

    cache.set(chunk_a, chunk_b, result)
    return result


# ---------------------------------------------------------------------------
# Double comparison (consensus)
# ---------------------------------------------------------------------------


async def compare_scale(
    client: openai.AsyncOpenAI,
    model: str,
    service_tier: str | None,
    chunk_a: str,
    chunk_b: str,
    cache: ComparisonCache,
    semaphore: asyncio.Semaphore,
) -> str:
    """Compare two chunks twice (a,b) and (b,a) for consensus.

    Returns 'a' if chunk_a is more abstract, 'b' if chunk_b is, or 'tie'
    if the two calls disagree.
    """
    r1, r2 = await asyncio.gather(
        _compare_once(client, model, service_tier, chunk_a, chunk_b, cache, semaphore),
        _compare_once(client, model, service_tier, chunk_b, chunk_a, cache, semaphore),
    )
    # r1: 'A'→chunk_a wins, 'B'→chunk_b wins
    # r2 was called with (b, a): 'A'→chunk_b wins (b was presented as A), 'B'→chunk_a wins
    winner1 = "a" if r1 == "A" else "b"
    winner2 = "b" if r2 == "A" else "a"
    if winner1 == winner2:
        return winner1
    return "tie"


# ---------------------------------------------------------------------------
# Active pair selection
# ---------------------------------------------------------------------------


def _select_next_pair(
    n_items: int,
    compared: set[frozenset],
    params: np.ndarray | None,
) -> tuple[int, int] | None:
    """Return the most informative (most uncertain) uncompared pair.

    Uses choix.probabilities to find the pair closest to P=0.5.
    Falls back to the first uncompared pair if params are unavailable.
    """
    candidates = [
        (i, j)
        for i in range(n_items)
        for j in range(i + 1, n_items)
        if frozenset([i, j]) not in compared
    ]
    if not candidates:
        return None
    if params is None:
        return candidates[0]

    best_pair = candidates[0]
    best_uncertainty = -1.0
    for i, j in candidates:
        p = choix.probabilities([i, j], params)[0]
        uncertainty = 1.0 - abs(2 * p - 1)  # 1.0 when p=0.5
        if uncertainty > best_uncertainty:
            best_uncertainty = uncertainty
            best_pair = (i, j)
    return best_pair


# ---------------------------------------------------------------------------
# Choix ranking
# ---------------------------------------------------------------------------


def compute_choix_params(n_items: int, data: list[tuple[int, int]]) -> np.ndarray:
    """Compute Bradley-Terry params via LSR. Higher = more abstract."""
    if not data or n_items < 2:
        return np.zeros(n_items)
    try:
        return choix.lsr_pairwise(n_items, data, alpha=0.1)
    except Exception:
        logger.warning("choix.lsr_pairwise failed for n=%d, data=%d", n_items, len(data))
        return np.zeros(n_items)


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------


async def run_problem_tournament(
    problem_id: str,
    field: str,
    chunks: list[str],
    client: openai.AsyncOpenAI,
    model: str,
    service_tier: str | None,
    semaphore: asyncio.Semaphore,
    cache: ComparisonCache,
    extra_comparisons: int,
) -> list[tuple[int, int]]:
    """Run a pairwise comparison tournament for a single problem-field.

    Phase 1: compare all consecutive pairs (i, i+1) in parallel.
    Phase 2: use choix active learning to pick the most informative
             remaining pairs, in batches of 5.

    Returns list of (winner_idx, loser_idx) for choix.
    """
    n = len(chunks)
    if n < 2:
        return []

    data: list[tuple[int, int]] = []
    compared: set[frozenset] = set()

    # Phase 1: consecutive pairs, all in parallel
    consecutive = [(i, i + 1) for i in range(n - 1)]
    results = await asyncio.gather(*[
        compare_scale(client, model, service_tier, chunks[i], chunks[j], cache, semaphore)
        for i, j in consecutive
    ])
    for (i, j), res in zip(consecutive, results):
        compared.add(frozenset([i, j]))
        if res == "a":
            data.append((i, j))
        elif res == "b":
            data.append((j, i))

    # Phase 2: active learning in batches of 5
    budget = extra_comparisons
    batch_size = 5
    while budget > 0:
        params = compute_choix_params(n, data) if data else None
        batch: list[tuple[int, int]] = []
        for _ in range(min(batch_size, budget)):
            pair = _select_next_pair(n, compared, params)
            if pair is None:
                break
            compared.add(frozenset(pair))
            batch.append(pair)
        if not batch:
            break

        batch_results = await asyncio.gather(*[
            compare_scale(client, model, service_tier, chunks[i], chunks[j], cache, semaphore)
            for i, j in batch
        ])
        for (i, j), res in zip(batch, batch_results):
            if res == "a":
                data.append((i, j))
            elif res == "b":
                data.append((j, i))
        budget -= len(batch)

    logger.debug(
        "%s [%s]: %d chunks, %d comparisons → %d wins recorded",
        problem_id, field, n, len(compared), len(data),
    )
    return data


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def rank_all(
    results: list[dict],
    config: dict,
    project_root: Path,
) -> list[dict]:
    """Run tournaments for all items and return chunk_ranking records.

    Each record:
      id, reasoning_chunks, answer_chunks,
      reasoning_params, answer_params,
      reasoning_comparisons, answer_comparisons
    """
    sr = config.get("slod_rank", {})
    sr_model = sr.get("model", {})
    model = sr_model.get("name", config["model"]["name"])
    service_tier = sr_model.get("service_tier")
    max_concurrent = sr.get("max_concurrent", 20)
    extra = sr.get("extra_comparisons_per_problem", 10)
    min_chunks = sr.get("min_chunks", 2)

    cache_path = (project_root / sr.get("cache_file", "../../data/sh6/comparison_cache.json")).resolve()
    cache = ComparisonCache(cache_path)
    cache.load()

    api_key_env = sr_model.get("api_key_env", config.get("model", {}).get("api_key_env"))
    base_url = sr_model.get("base_url", config.get("model", {}).get("base_url"))
    api_key = os.environ.get(api_key_env) if api_key_env else os.environ.get("OPENAI_API_KEY")
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _rank_item(item: dict) -> dict | None:
        if item.get("error"):
            return None
        iid = item["id"]
        r_chunks = chunk_text(item.get("reasoning_text") or "")
        a_chunks = chunk_text(item.get("answer_text") or "")

        if len(r_chunks) < min_chunks and len(a_chunks) < min_chunks:
            logger.debug("Skipping %s: insufficient chunks", iid)
            return None

        r_data, a_data = await asyncio.gather(
            run_problem_tournament(iid, "reasoning", r_chunks, client, model, service_tier, semaphore, cache, extra),
            run_problem_tournament(iid, "answer", a_chunks, client, model, service_tier, semaphore, cache, extra),
        )

        r_params = compute_choix_params(len(r_chunks), r_data).tolist()
        a_params = compute_choix_params(len(a_chunks), a_data).tolist()

        return {
            "id": iid,
            "reasoning_chunks": r_chunks,
            "answer_chunks": a_chunks,
            "reasoning_params": r_params,
            "answer_params": a_params,
            "reasoning_comparisons": [list(pair) for pair in r_data],
            "answer_comparisons": [list(pair) for pair in a_data],
        }

    tasks = [_rank_item(item) for item in results]
    rankings: list[dict] = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        rec = await coro
        completed += 1
        if rec is not None:
            rankings.append(rec)
        if completed % 20 == 0 or completed == len(tasks):
            logger.info("Tournament progress: %d/%d items processed", completed, len(tasks))

    cache.save()
    return rankings
