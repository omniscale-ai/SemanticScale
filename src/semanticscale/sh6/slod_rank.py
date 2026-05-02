"""SLoD pairwise ranking for SH6 reasoning trajectories.

Chunks texts by \\n\\n, runs double LLM comparisons (a→b, b→a) for
consensus, caches directional results to disk, and uses OpenSkill's
Plackett-Luce model to compute abstraction scores (higher = more abstract).

Dispatches between OpenAI Responses API and OpenRouter Chat Completions
via :mod:`semanticscale.llm_backend`.
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import openai
from openskill.models import PlackettLuce
from pydantic import BaseModel

from semanticscale.llm_backend import Backend, make_backend

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


@dataclass(frozen=True)
class PairMatch:
    """One observed pairwise outcome on the local SLoD scale."""

    left: int
    right: int
    outcome: Literal["a", "b", "tie"]


@dataclass(frozen=True)
class RankingState:
    """Current OpenSkill fit used for scoring and active pair selection."""

    model: PlackettLuce
    ratings: list[list[object]]
    params: np.ndarray


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str) -> list[str]:
    """Split text by \\n\\n and return non-empty stripped chunks."""
    return [c.strip() for c in text.split("\n\n") if c.strip()]


def coalesce_chunks(chunks: list[str], max_chunks: int | None) -> list[str]:
    """Merge adjacent chunks into at most max_chunks ordered groups.

    Free-form models can emit hundreds of blank-line-separated paragraphs.
    Coalescing preserves order and text while bounding pairwise tournament
    cost for configs that opt into it.
    """
    if max_chunks is None or max_chunks <= 0 or len(chunks) <= max_chunks:
        return chunks
    n_chunks = len(chunks)
    merged = []
    for i in range(max_chunks):
        start = i * n_chunks // max_chunks
        end = (i + 1) * n_chunks // max_chunks
        merged.append("\n\n".join(chunks[start:end]))
    return merged


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
    backend: Backend,
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

    async with semaphore:
        parsed: AbstractionWinner = await backend.parse(
            model=model,
            messages=messages,
            text_format=AbstractionWinner,
            service_tier=service_tier,
        )

    result = parsed.winner
    cache.set(chunk_a, chunk_b, result)
    return result


# ---------------------------------------------------------------------------
# Double comparison (consensus)
# ---------------------------------------------------------------------------


def _is_context_overflow(exc: BaseException) -> bool:
    """True when the judge rejected a pair as exceeding its context window."""
    if not isinstance(exc, openai.BadRequestError):
        return False
    msg = str(getattr(exc, "message", "") or exc)
    return "maximum context length" in msg or "context length" in msg.lower()


async def compare_scale(
    backend: Backend,
    model: str,
    service_tier: str | None,
    chunk_a: str,
    chunk_b: str,
    cache: ComparisonCache,
    semaphore: asyncio.Semaphore,
) -> str:
    """Compare two chunks twice (a,b) and (b,a) for consensus.

    Returns 'a' if chunk_a is more abstract, 'b' if chunk_b is, or 'tie'
    if the two calls disagree. If either direction overflows the judge's
    context window, the pair is reported as a tie so a single oversize
    chunk doesn't kill the whole tournament.
    """
    try:
        r1, r2 = await asyncio.gather(
            _compare_once(backend, model, service_tier, chunk_a, chunk_b, cache, semaphore),
            _compare_once(backend, model, service_tier, chunk_b, chunk_a, cache, semaphore),
        )
    except openai.BadRequestError as exc:
        if _is_context_overflow(exc):
            logger.warning("Pair exceeds judge context; reporting tie. (%s)", exc)
            return "tie"
        raise
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
    state: RankingState | None,
) -> tuple[int, int] | None:
    """Return the most informative (most uncertain) uncompared pair.

    Uses the current OpenSkill fit to score the entropy of win / loss / tie
    outcomes for each remaining pair.
    Falls back to the first uncompared pair if no fit is available yet.
    """
    candidates = [
        (i, j)
        for i in range(n_items)
        for j in range(i + 1, n_items)
        if frozenset([i, j]) not in compared
    ]
    if not candidates:
        return None
    if state is None:
        return candidates[0]

    best_pair = candidates[0]
    best_uncertainty = -1.0
    for i, j in candidates:
        teams = [state.ratings[i], state.ratings[j]]
        draw_prob = float(state.model.predict_draw(teams))
        win_probs = np.asarray(state.model.predict_win(teams), dtype=float)
        probs = np.array(
            [
                (1.0 - draw_prob) * win_probs[0],
                (1.0 - draw_prob) * win_probs[1],
                draw_prob,
            ],
            dtype=float,
        )
        probs /= max(float(probs.sum()), 1e-12)
        uncertainty = float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum() / np.log(3.0))
        if uncertainty > best_uncertainty:
            best_uncertainty = uncertainty
            best_pair = (i, j)
    return best_pair


# ---------------------------------------------------------------------------
# OpenSkill ranking
# ---------------------------------------------------------------------------


def _match_ranks(match: PairMatch) -> list[int]:
    if match.outcome == "a":
        return [0, 1]
    if match.outcome == "b":
        return [1, 0]
    return [0, 0]


def _fit_openskill_order(n_items: int, matches: list[PairMatch]) -> tuple[PlackettLuce, list[list[object]]]:
    model = PlackettLuce()
    ratings = [[model.rating(name=str(i))] for i in range(n_items)]
    for match in matches:
        left = ratings[match.left]
        right = ratings[match.right]
        updated = model.rate([left, right], ranks=_match_ranks(match))
        ratings[match.left], ratings[match.right] = updated
    return model, ratings


def _average_ratings(
    model: PlackettLuce,
    forward: list[list[object]],
    reverse: list[list[object]],
) -> list[list[object]]:
    averaged: list[list[object]] = []
    for idx, (left_team, right_team) in enumerate(zip(forward, reverse, strict=True)):
        left = left_team[0]
        right = right_team[0]
        averaged.append([
            model.rating(
                mu=float((left.mu + right.mu) / 2.0),
                sigma=float((left.sigma + right.sigma) / 2.0),
                name=str(idx),
            )
        ])
    return averaged


def compute_ranking_state(n_items: int, matches: list[PairMatch]) -> RankingState | None:
    """Fit OpenSkill scores from pairwise wins/losses/ties."""
    if not matches or n_items < 2:
        return None
    try:
        _, forward = _fit_openskill_order(n_items, matches)
        model, reverse = _fit_openskill_order(n_items, list(reversed(matches)))
        ratings = _average_ratings(model, forward, reverse)
        params = np.array([float(team[0].mu) for team in ratings], dtype=float)
        params -= float(params.mean())
        return RankingState(model=model, ratings=ratings, params=params)
    except Exception:
        logger.warning("OpenSkill fit failed for n=%d, matches=%d", n_items, len(matches))
        return None


def compute_openskill_params(n_items: int, matches: list[PairMatch]) -> np.ndarray:
    """Compute mean-centered OpenSkill scores. Higher = more abstract."""
    state = compute_ranking_state(n_items, matches)
    if state is None:
        return np.zeros(n_items)
    return state.params


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------


async def run_problem_tournament(
    problem_id: str,
    field: str,
    chunks: list[str],
    backend: Backend,
    model: str,
    service_tier: str | None,
    semaphore: asyncio.Semaphore,
    cache: ComparisonCache,
    extra_comparisons: int,
) -> list[PairMatch]:
    """Run a pairwise comparison tournament for a single problem-field.

    Phase 1: compare all consecutive pairs (i, i+1) in parallel.
    Phase 2: use OpenSkill active learning to pick the most informative
             remaining pairs, in batches of 5.
    """
    n = len(chunks)
    if n < 2:
        return []

    matches: list[PairMatch] = []
    compared: set[frozenset] = set()

    # Phase 1: consecutive pairs, all in parallel
    consecutive = [(i, i + 1) for i in range(n - 1)]
    results = await asyncio.gather(*[
        compare_scale(backend, model, service_tier, chunks[i], chunks[j], cache, semaphore)
        for i, j in consecutive
    ])
    for (i, j), res in zip(consecutive, results):
        compared.add(frozenset([i, j]))
        matches.append(PairMatch(i, j, res))

    # Phase 2: active learning in batches of 5
    budget = extra_comparisons
    batch_size = 5
    while budget > 0:
        state = compute_ranking_state(n, matches)
        batch: list[tuple[int, int]] = []
        for _ in range(min(batch_size, budget)):
            pair = _select_next_pair(n, compared, state)
            if pair is None:
                break
            compared.add(frozenset(pair))
            batch.append(pair)
        if not batch:
            break

        batch_results = await asyncio.gather(*[
            compare_scale(backend, model, service_tier, chunks[i], chunks[j], cache, semaphore)
            for i, j in batch
        ])
        for (i, j), res in zip(batch, batch_results):
            matches.append(PairMatch(i, j, res))
        budget -= len(batch)

    n_wins = sum(1 for match in matches if match.outcome != "tie")
    n_ties = len(matches) - n_wins
    logger.debug(
        "%s [%s]: %d chunks, %d comparisons → %d decisive, %d ties",
        problem_id, field, n, len(compared), n_wins, n_ties,
    )
    return matches


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def rank_all(
    results: list[dict],
    config: dict,
    run_dir: Path,
) -> list[dict]:
    """Run tournaments for all items and return chunk_ranking records.

    The comparison cache is stored at ``run_dir / "comparison_cache.json"``.

    Each record:
      id, reasoning_chunks, answer_chunks,
      reasoning_params, answer_params,
      reasoning_comparisons, answer_comparisons,
      reasoning_ties, answer_ties
    """
    sr = config["pairwise_slod"]
    sr_model = sr["model"]
    model = sr_model["name"]
    service_tier = sr_model.get("service_tier")
    max_concurrent = sr.get("max_concurrent", 20)
    extra = sr.get("extra_comparisons_per_problem", 10)
    min_chunks = sr.get("min_chunks", 2)

    cache_path = run_dir / "comparison_cache.json"
    cache = ComparisonCache(cache_path)
    cache.load()

    backend = make_backend(sr_model)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _rank_item(item: dict) -> dict | None:
        if item.get("error"):
            return None
        iid = item["id"]
        # Prefer pre-chunked traces (e.g. ProcessBench steps); fall back to
        # splitting free-form text by \n\n.
        pre_r = item.get("reasoning_chunks")
        pre_a = item.get("answer_chunks")
        r_chunks = list(pre_r) if pre_r else chunk_text(item.get("reasoning_text") or "")
        a_chunks = list(pre_a) if pre_a else chunk_text(item.get("answer_text") or "")
        r_chunks = coalesce_chunks(r_chunks, sr.get("max_reasoning_chunks"))
        a_chunks = coalesce_chunks(a_chunks, sr.get("max_answer_chunks"))

        n_r = len(r_chunks)
        n_a = len(a_chunks)
        if n_r + n_a < min_chunks:
            logger.debug("Skipping %s: insufficient chunks (%d)", iid, n_r + n_a)
            return None

        # Joint tournament so reasoning and answer share a common SLoD scale.
        combined = r_chunks + a_chunks
        combined_matches = await run_problem_tournament(
            iid, "combined", combined, backend, model, service_tier, semaphore, cache, extra,
        )
        joint_params = compute_openskill_params(len(combined), combined_matches).tolist()
        r_params = joint_params[:n_r]
        a_params = joint_params[n_r:]
        r_data = [
            [match.left, match.right]
            if match.outcome == "a"
            else [match.right, match.left]
            for match in combined_matches
            if match.outcome != "tie" and match.left < n_r and match.right < n_r
        ]
        a_data = [
            [match.left - n_r, match.right - n_r]
            if match.outcome == "a"
            else [match.right - n_r, match.left - n_r]
            for match in combined_matches
            if match.outcome != "tie" and match.left >= n_r and match.right >= n_r
        ]
        r_ties = [
            [match.left, match.right]
            for match in combined_matches
            if match.outcome == "tie" and match.left < n_r and match.right < n_r
        ]
        a_ties = [
            [match.left - n_r, match.right - n_r]
            for match in combined_matches
            if match.outcome == "tie" and match.left >= n_r and match.right >= n_r
        ]

        return {
            "id": iid,
            "reasoning_chunks": r_chunks,
            "answer_chunks": a_chunks,
            "reasoning_params": r_params,
            "answer_params": a_params,
            "reasoning_comparisons": r_data,
            "answer_comparisons": a_data,
            "reasoning_ties": r_ties,
            "answer_ties": a_ties,
        }

    try:
        tasks = [_rank_item(item) for item in results]
        rankings: list[dict] = []
        completed = 0
        for coro in asyncio.as_completed(tasks):
            try:
                rec = await coro
            except Exception:
                # One item's tournament blew up. Log and move on so the
                # remaining 100+ in-flight tournaments aren't taken down
                # by aclose() in the finally block.
                logger.exception("Item ranking failed; skipping")
                rec = None
            completed += 1
            if rec is not None:
                rankings.append(rec)
            if completed % 20 == 0 or completed == len(tasks):
                logger.info("Tournament progress: %d/%d items processed", completed, len(tasks))
                cache.save()
    finally:
        cache.save()
        await backend.aclose()

    return rankings
