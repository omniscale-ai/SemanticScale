#!/usr/bin/env python
"""SH6 — Phase A: anchor-based absolute SLoD validation.

Diagnostic for the hypothesis that processbench/agenthallu fail SLoD-based
failure prediction (ROC-AUC ~0.50) because their traces genuinely lack
absolute variance on the macro↔micro axis. SH6's current per-trace
mean-centred OpenSkill ratings cannot distinguish a homogeneous trace from
one spanning levels, since ratings carry no shared unit across traces.

Approach (no pipeline changes yet):

  Step 1 — anchor self-consistency: run all-pairs LLM comparisons among a
           fixed set of 15 hand-tiered anchor chunks (macro/meso/micro),
           fit OpenSkill, check Spearman ρ between fitted μ and hand tier.
           Fails fast if the anchor set is incoherent.

  Step 2 — chunk-vs-anchor scoring: sample chunks from each dataset's
           existing chunk_rankings.jsonl, run chunk↔anchor pairwise
           comparisons, fit one OpenSkill over {anchors ∪ sampled_chunks}
           so every chunk lands on the same axis as the anchors. No
           mean-centring.

  Step 3 — distribution summary: per-dataset histogram of absolute μ,
           per-trace std distribution. The hypothesis predicts that
           processbench/agenthallu have narrower absolute-μ spreads and
           smaller per-trace stds than swe-agent/frontierscience.

Outputs → experiments/sh6_llm-pairwise-slod/reports/anchor_validation/:
  - anchor_consistency.json
  - absolute_slod.jsonl       (per chunk: dataset, trace_id, field, anchor_mu)
  - per_dataset_summary.csv
  - distributions.png
  - summary.md

Usage:
    uv run --env-file .env python experiments/sh6_llm-pairwise-slod/scripts/06_anchor_validation.py \\
        --n-traces-per-dataset 30 --max-chunks-per-trace 8

    # Smoke test (one LLM call per pair, trivial sample):
    uv run --env-file .env python experiments/sh6_llm-pairwise-slod/scripts/06_anchor_validation.py \\
        --anchor-consistency-only

    uv run --env-file .env python experiments/sh6_llm-pairwise-slod/scripts/06_anchor_validation.py \\
        --n-traces-per-dataset 3 --max-chunks-per-trace 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from semanticscale.llm_backend import make_backend
from semanticscale.sh6.slod_rank import (
    ComparisonCache,
    PairMatch,
    _fit_openskill_order,
    _average_ratings,
    compare_scale,
)
from semanticscale.utils import load_jsonl, setup_logging

logger = logging.getLogger(__name__)


HERE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = HERE
DATA_ROOT = (HERE / "../../data/sh6").resolve()
ANCHORS_PATH = HERE / "anchors" / "slod_anchors.yaml"
OUT_DIR = HERE / "reports" / "anchor_validation"

# Which (dataset, run_slug) pairs to sample from. These are the four runs
# that currently have chunk_rankings.jsonl and for which failure_prediction
# has been reported.
DATASETS = [
    ("processbench", "gsm8k"),
    ("agenthallu", "framework-all"),
    ("swe-agent-trajectories", "model-all"),
    ("frontierscience", "deepseek/deepseek-v3.2_reasoning-auto"),
]


# --------------------------------------------------------------------------- #
# Anchors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Anchor:
    id: str
    tier: int   # 3 = macro, 2 = meso, 1 = micro
    text: str


def load_anchors(path: Path = ANCHORS_PATH) -> list[Anchor]:
    with path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    anchors = [Anchor(id=a["id"], tier=int(a["tier"]), text=a["text"].strip())
               for a in doc["anchors"]]
    logger.info("Loaded %d anchors (version=%s)", len(anchors), doc.get("version"))
    return anchors


# --------------------------------------------------------------------------- #
# Chunk sampling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Chunk:
    dataset: str
    run_slug: str
    trace_id: str
    field: str   # "reasoning" | "answer"
    chunk_idx: int
    text: str


def sample_chunks(
    dataset: str,
    run_slug: str,
    n_traces: int,
    max_chunks_per_trace: int,
    rng: random.Random,
) -> list[Chunk]:
    """Sample chunks from an existing chunk_rankings.jsonl file.

    For each trace, take up to `max_chunks_per_trace` chunks across the
    reasoning+answer fields. Traces are sampled uniformly without
    replacement from the full file; chunk subsampling within a trace is
    uniform if the trace is longer than the cap.
    """
    path = DATA_ROOT / dataset / run_slug / "chunk_rankings.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"chunk_rankings.jsonl missing: {path}")
    traces = load_jsonl(path)
    rng.shuffle(traces)
    traces = traces[:n_traces]

    chunks: list[Chunk] = []
    for trace in traces:
        tid = trace["id"]
        pool: list[tuple[str, int, str]] = []
        for field in ("reasoning", "answer"):
            raw = trace.get(f"{field}_chunks") or []
            for idx, text in enumerate(raw):
                txt = (text or "").strip()
                if txt:
                    pool.append((field, idx, txt))
        if not pool:
            continue
        if len(pool) > max_chunks_per_trace:
            pool = rng.sample(pool, max_chunks_per_trace)
        for field, idx, txt in pool:
            chunks.append(Chunk(
                dataset=dataset,
                run_slug=run_slug,
                trace_id=tid,
                field=field,
                chunk_idx=idx,
                text=txt,
            ))
    logger.info(
        "Sampled %d chunks from %s/%s (%d traces)",
        len(chunks), dataset, run_slug, len(traces),
    )
    return chunks


# --------------------------------------------------------------------------- #
# Pairwise comparison runner
# --------------------------------------------------------------------------- #


async def _run_comparisons(
    pairs: list[tuple[int, int]],
    texts: list[str],
    backend,
    model: str,
    service_tier: str | None,
    cache: ComparisonCache,
    semaphore: asyncio.Semaphore,
    batch_size: int = 200,
) -> list[PairMatch]:
    """Run `compare_scale` on every (i, j) pair, batched and fault-tolerant.

    Pairs are processed in batches; the cache is flushed after each batch so
    crashes don't lose progress. Per-pair failures (e.g. transient OpenRouter
    response-validation errors) are logged and dropped — OpenSkill handles a
    sparse match set without trouble.
    """
    matches: list[PairMatch] = []
    n_failed = 0
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        tasks = [
            compare_scale(backend, model, service_tier, texts[i], texts[j], cache, semaphore)
            for i, j in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (i, j), res in zip(batch, results):
            if isinstance(res, BaseException):
                n_failed += 1
                logger.warning("compare_scale failed for pair (%d,%d): %s",
                               i, j, type(res).__name__)
                continue
            matches.append(PairMatch(i, j, res))
        cache.save()
        logger.info("Comparisons progress: %d/%d pairs (%d failed so far)",
                    min(start + batch_size, len(pairs)), len(pairs), n_failed)
    if n_failed:
        logger.warning("Total failed comparisons: %d / %d (%.1f%%)",
                       n_failed, len(pairs), 100.0 * n_failed / max(len(pairs), 1))
    return matches


def fit_openskill_absolute(
    n_items: int,
    matches: list[PairMatch],
) -> np.ndarray:
    """Fit OpenSkill on all pairwise matches and return un-centred μ.

    Mirrors `slod_rank.compute_openskill_params` but skips the per-pool
    mean-centring. The resulting μ values live on the anchor-defined axis,
    so they are comparable across chunks, traces, and datasets.
    """
    if not matches or n_items < 2:
        return np.zeros(n_items)
    _, forward = _fit_openskill_order(n_items, matches)
    model, reverse = _fit_openskill_order(n_items, list(reversed(matches)))
    ratings = _average_ratings(model, forward, reverse)
    return np.array([float(team[0].mu) for team in ratings], dtype=float)


# --------------------------------------------------------------------------- #
# Step 1: anchor self-consistency
# --------------------------------------------------------------------------- #


async def step1_anchor_consistency(
    anchors: list[Anchor],
    backend,
    model: str,
    service_tier: str | None,
    cache: ComparisonCache,
    semaphore: asyncio.Semaphore,
) -> dict:
    logger.info("Step 1: anchor self-consistency (%d anchors, %d pairs)",
                len(anchors), len(anchors) * (len(anchors) - 1) // 2)
    texts = [a.text for a in anchors]
    pairs = [(i, j) for i in range(len(anchors)) for j in range(i + 1, len(anchors))]
    matches = await _run_comparisons(pairs, texts, backend, model, service_tier, cache, semaphore)

    mu = fit_openskill_absolute(len(anchors), matches)
    tiers = np.array([a.tier for a in anchors])
    rho, pval = spearmanr(mu, tiers)

    n_ties = sum(1 for m in matches if m.outcome == "tie")
    result = {
        "n_anchors": len(anchors),
        "n_pairs": len(pairs),
        "n_ties": n_ties,
        "spearman_rho": float(rho),
        "spearman_p": float(pval),
        "anchors": [
            {"id": a.id, "tier": a.tier, "mu": float(mu[i]),
             "text_preview": a.text[:100]}
            for i, a in enumerate(anchors)
        ],
    }
    logger.info("Spearman ρ(μ, tier) = %.3f (p=%.3g), ties=%d/%d",
                rho, pval, n_ties, len(pairs))
    return result


# --------------------------------------------------------------------------- #
# Step 2: chunk-vs-anchor joint fit
# --------------------------------------------------------------------------- #


async def step2_chunk_anchor_fit(
    anchors: list[Anchor],
    chunks: list[Chunk],
    backend,
    model: str,
    service_tier: str | None,
    cache: ComparisonCache,
    semaphore: asyncio.Semaphore,
) -> tuple[np.ndarray, np.ndarray]:
    """Run every (chunk, anchor) pair and fit joint OpenSkill.

    Returns (anchor_mu, chunk_mu) on the same absolute scale.
    """
    n_a = len(anchors)
    n_c = len(chunks)
    texts = [a.text for a in anchors] + [c.text for c in chunks]
    n_total = n_a + n_c

    # Every chunk paired with every anchor (index offsets: anchors 0..n_a-1,
    # chunks n_a..n_a+n_c-1). No chunk-chunk pairs in this diagnostic pass.
    pairs = [(i, n_a + j) for j in range(n_c) for i in range(n_a)]
    logger.info("Step 2: %d chunks × %d anchors = %d pairs", n_c, n_a, len(pairs))

    matches = await _run_comparisons(pairs, texts, backend, model, service_tier, cache, semaphore)
    mu = fit_openskill_absolute(n_total, matches)
    return mu[:n_a], mu[n_a:]


# --------------------------------------------------------------------------- #
# Step 3: per-dataset summary + plot
# --------------------------------------------------------------------------- #


def summarise(chunks: list[Chunk], chunk_mu: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({
        "dataset": [c.dataset for c in chunks],
        "trace_id": [c.trace_id for c in chunks],
        "field": [c.field for c in chunks],
        "chunk_idx": [c.chunk_idx for c in chunks],
        "anchor_mu": chunk_mu,
    })
    return df


def per_dataset_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in df.groupby("dataset"):
        per_trace_std = sub.groupby("trace_id")["anchor_mu"].std(ddof=0)
        per_trace_range = sub.groupby("trace_id")["anchor_mu"].agg(lambda x: x.max() - x.min())
        rows.append({
            "dataset": dataset,
            "n_chunks": len(sub),
            "n_traces": sub["trace_id"].nunique(),
            "mu_mean": float(sub["anchor_mu"].mean()),
            "mu_std": float(sub["anchor_mu"].std(ddof=0)),
            "mu_min": float(sub["anchor_mu"].min()),
            "mu_max": float(sub["anchor_mu"].max()),
            "per_trace_std_median": float(per_trace_std.median()),
            "per_trace_std_mean": float(per_trace_std.mean()),
            "per_trace_range_median": float(per_trace_range.median()),
            "per_trace_range_mean": float(per_trace_range.mean()),
        })
    return pd.DataFrame(rows).set_index("dataset")


def plot_distributions(
    df: pd.DataFrame,
    anchor_mu: np.ndarray,
    anchors: list[Anchor],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram per dataset of absolute μ, with anchor tier bands overlaid
    ax = axes[0]
    for dataset in datasets:
        vals = df.loc[df["dataset"] == dataset, "anchor_mu"].values
        ax.hist(vals, bins=25, alpha=0.45, label=f"{dataset} (n={len(vals)})", density=True)

    for tier, color, label in [(3, "C3", "macro"), (2, "C1", "meso"), (1, "C2", "micro")]:
        tier_vals = [anchor_mu[i] for i, a in enumerate(anchors) if a.tier == tier]
        if tier_vals:
            ax.axvspan(min(tier_vals), max(tier_vals), color=color, alpha=0.08)
            for v in tier_vals:
                ax.axvline(v, color=color, linestyle=":", alpha=0.6, linewidth=0.8)
            ax.text(np.mean(tier_vals), ax.get_ylim()[1] * 0.95, label,
                    color=color, ha="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("absolute SLoD μ (anchor-calibrated)")
    ax.set_ylabel("density")
    ax.set_title("Per-dataset distribution of absolute μ")
    ax.legend(fontsize=8, loc="upper right")

    # Right: per-trace std boxplot
    ax = axes[1]
    trace_stds = [
        df[df["dataset"] == ds].groupby("trace_id")["anchor_mu"].std(ddof=0).dropna().values
        for ds in datasets
    ]
    ax.boxplot(trace_stds, labels=datasets, showmeans=True)
    ax.set_ylabel("per-trace std of absolute μ")
    ax.set_title("Within-trace spread (genuine absolute variance)")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("SH6 anchor-based absolute SLoD — Phase A diagnostic")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote distribution plot → %s", out_path)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table without `tabulate`."""
    cols = list(df.columns)
    header = "| " + " | ".join([df.index.name or ""] + cols) + " |"
    sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
    lines = [header, sep]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [
            f"{v:.3f}" if isinstance(v, (int, float)) and not pd.isna(v) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_md(
    out_path: Path,
    anchor_consistency: dict,
    stats: pd.DataFrame | None,
    n_pairs_step2: int | None,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# SH6 Phase A — anchor-based absolute SLoD validation",
        "",
        "## Anchor self-consistency (Step 1)",
        "",
        f"- n_anchors: {anchor_consistency['n_anchors']}",
        f"- n_pairs: {anchor_consistency['n_pairs']}",
        f"- n_ties: {anchor_consistency['n_ties']}",
        f"- **Spearman ρ(μ, tier) = {anchor_consistency['spearman_rho']:.3f}**"
        f"  (p = {anchor_consistency['spearman_p']:.3g})",
        "",
        "| anchor | tier | μ |",
        "|---|---|---|",
    ]
    for a in sorted(anchor_consistency["anchors"], key=lambda x: -x["mu"]):
        lines.append(f"| {a['id']} | {a['tier']} | {a['mu']:.2f} |")
    lines.append("")

    if stats is not None:
        lines += [
            "## Per-dataset absolute SLoD (Step 2)",
            "",
            f"- Pairs compared: {n_pairs_step2}",
            f"- Traces sampled per dataset: {args.n_traces_per_dataset}",
            f"- Max chunks per trace: {args.max_chunks_per_trace}",
            "",
            _df_to_markdown(stats.round(3)),
            "",
            "### Interpretation",
            "",
            "Hypothesis: processbench/agenthallu per-trace std should be "
            "meaningfully smaller (≥40%) than swe-agent/frontierscience if "
            "their traces genuinely lack absolute SLoD variance.",
            "",
        ]
    else:
        lines += [
            "## Step 2 skipped",
            "",
            "Run without `--anchor-consistency-only` to collect per-dataset distributions.",
            "",
        ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote summary → %s", out_path)


# --------------------------------------------------------------------------- #
# Config loading (for LLM backend)
# --------------------------------------------------------------------------- #


def load_llm_settings(config_path: Path) -> tuple[dict, int]:
    """Return (model_cfg, max_concurrent) from an SH6 config.yaml.

    We reuse an existing config (default: processbench) to avoid duplicating
    backend settings. Only the `pairwise_slod.model` and `max_concurrent`
    fields are used here; the rest of the config is ignored.
    """
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sr = cfg["pairwise_slod"]
    return sr["model"], int(sr.get("max_concurrent", 20))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


async def _async_main(args: argparse.Namespace) -> None:
    anchors = load_anchors()

    model_cfg, max_concurrent = load_llm_settings(Path(args.llm_config))
    backend = make_backend(model_cfg)
    model = model_cfg["name"]
    service_tier = model_cfg.get("service_tier")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "comparison_cache.json"
    cache = ComparisonCache(cache_path)
    cache.load()
    semaphore = asyncio.Semaphore(max_concurrent)

    try:
        # Step 1 — anchor self-consistency
        anchor_consistency = await step1_anchor_consistency(
            anchors, backend, model, service_tier, cache, semaphore,
        )
        cache.save()
        (OUT_DIR / "anchor_consistency.json").write_text(
            json.dumps(anchor_consistency, indent=2), encoding="utf-8",
        )

        rho = anchor_consistency["spearman_rho"]
        if rho < args.anchor_rho_threshold:
            logger.warning(
                "Anchor Spearman ρ=%.3f below threshold %.2f — anchors may be "
                "incoherent. Refine the anchor set before trusting Step 2.",
                rho, args.anchor_rho_threshold,
            )
            if not args.force:
                write_summary_md(OUT_DIR / "summary.md", anchor_consistency,
                                 None, None, args)
                return

        if args.anchor_consistency_only:
            write_summary_md(OUT_DIR / "summary.md", anchor_consistency,
                             None, None, args)
            return

        # Step 2 — sample chunks per dataset, run joint OpenSkill
        rng = random.Random(args.seed)
        all_chunks: list[Chunk] = []
        for dataset, run_slug in DATASETS:
            all_chunks.extend(sample_chunks(
                dataset, run_slug,
                args.n_traces_per_dataset,
                args.max_chunks_per_trace,
                rng,
            ))
        logger.info("Total sampled chunks across datasets: %d", len(all_chunks))

        anchor_mu2, chunk_mu = await step2_chunk_anchor_fit(
            anchors, all_chunks, backend, model, service_tier, cache, semaphore,
        )
        cache.save()

        # Step 3 — aggregate + plot
        df = summarise(all_chunks, chunk_mu)
        df.to_json(OUT_DIR / "absolute_slod.jsonl", orient="records", lines=True)

        stats = per_dataset_stats(df)
        stats.to_csv(OUT_DIR / "per_dataset_summary.csv")
        logger.info("Per-dataset stats:\n%s", stats.round(3).to_string())

        plot_distributions(df, anchor_mu2, anchors, OUT_DIR / "distributions.png")
        write_summary_md(
            OUT_DIR / "summary.md",
            anchor_consistency,
            stats,
            n_pairs_step2=len(anchors) * len(all_chunks),
            args=args,
        )
    finally:
        cache.save()
        await backend.aclose()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--llm-config",
        default=str(HERE / "config-processbench.yaml"),
        help=(
            "SH6 config to read `pairwise_slod.model` + `max_concurrent` from. "
            "Dataset settings inside the config are ignored."
        ),
    )
    parser.add_argument("--n-traces-per-dataset", type=int, default=30)
    parser.add_argument("--max-chunks-per-trace", type=int, default=8)
    parser.add_argument(
        "--anchor-rho-threshold",
        type=float,
        default=0.80,
        help="Minimum Spearman ρ(μ, tier) before Step 2 runs (safety gate).",
    )
    parser.add_argument(
        "--anchor-consistency-only",
        action="store_true",
        help="Run only Step 1 and exit (cheap smoke test).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue to Step 2 even if anchor consistency is below threshold.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
