"""Render the 3-panel graphical-abstract teaser used in the paper.

Panels:
  (A) Schematic: a four-step reasoning trace projecting onto the SLoD axis.
  (B) Concrete real failure case: SWE-agent ReproNim/reproman-518, llama-70B.
      Loaded from chunk_rankings.jsonl when available; otherwise the hardcoded
      values below (extracted from the HTML illustration on origin/main) are used.
  (C) Pass@1 lift bar chart on the FrontierScience Olympiad subset.

Usage:
    python experiments/sh6_llm-pairwise-slod/scripts/_make_teaser_figure.py \
        [--out SLoD-ICML-2026-Workshop/figures/fig_teaser_real.png] \
        [--rankings path/to/chunk_rankings.jsonl] [--id <case-id>]

This writes the composite teaser PNG and three standalone panel PDFs next to it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox

# === Style ===
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

CORRECT = "#2b6cb0"   # blue, used for "+ macro / correct"
WRONG = "#dd6b20"     # orange (reserved; not used directly in the real-data B)
NEUTRAL = "#4a5568"   # gray for axes and body text
LIGHT = "#cbd5e0"     # light gray
MACRO_C = "#1f77b4"   # matplotlib default blue, matches SH6 illustrations
MICRO_C = "#d62728"   # matplotlib default red, matches SH6 illustrations
FLIP_C = "#f59e0b"    # orange for sign-flip edges

# === Hardcoded fallback for Panel B ===
# SWE-agent · ReproNim__reproman-518 · llama-70B · WRONG, exit_context.
# Source: experiments/sh6_llm-pairwise-slod/reports/_cross_dataset/illustrations/
#   swe_ReproNim_reproman_518_search_loop.html (origin/main).
DEFAULT_CASE_ID = "ReproNim__reproman-518"
DEFAULT_BT_SCORES = [6.43, -4.27, 2.14, -4.30, 0.00]
DEFAULT_CHUNK_SNIPPETS = [
    'search_dir "run" src',
    'search_dir "run"  (no src/)',
    'search_dir "run" --name "*.py"',
    'search_dir "run" --name "*.py"',
    'grep -r "run" .',
]
DEFAULT_CASE_TITLE = ""

MANUAL_REASONING_SUMMARIES = {
    DEFAULT_CASE_ID: [
        "targeted search in missing src/",
        "broad repo search with no focus",
        "invalid filtered search (.py flag)",
        "repeat same invalid search",
        "panic fallback: grep entire repo",
    ],
}


def load_case_from_rankings(rankings_path: Path, case_id: str) -> tuple[list[float], list[str]]:
    """Load (bt_scores, chunk_snippets) for ``case_id`` from a chunk_rankings.jsonl."""
    with rankings_path.open() as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("id") == case_id:
                rp = d.get("reasoning_params") or []
                rc = d.get("reasoning_chunks") or []
                return list(map(float, rp)), [c.replace("\n", " ").strip() for c in rc]
    raise SystemExit(f"id {case_id} not found in {rankings_path}")


def _normalize_chunk_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split())


def _summarize_generic_chunk(text: str, index: int, previous_text: str | None = None) -> str:
    """Compress a raw chunk into a short label that preserves its SLoD role."""
    normalized = _normalize_chunk_text(text)
    lowered = normalized.lower()
    previous_normalized = _normalize_chunk_text(previous_text) if previous_text else None

    if previous_normalized and normalized == previous_normalized:
        return "repeat previous move"

    if lowered.startswith("search_dir"):
        if " src" in lowered:
            return "targeted search in missing src/"
        if "--name" in lowered or "*.py" in lowered:
            return "invalid filtered search"
        if lowered.rstrip().endswith('"run"') or lowered == "search_dir \"run\"":
            return "broad repo search with no focus"
        return "search step"

    if lowered.startswith("grep -r") or lowered.startswith("rg ") or lowered.startswith("grep "):
        if lowered.startswith("grep -r") and lowered.endswith(" ."):
            return "panic fallback: grep entire repo"
        return "repo-wide text scan"

    if any(token in lowered for token in ("error", "failed", "syntax", "invalid flag", "not found")):
        return "error feedback / tool failure"

    if any(token in lowered for token in ("open ", "cat ", "sed ", "read_file", "view ")):
        return "inspect file contents"

    if any(token in lowered for token in ("edit", "patch", "write", "replace", "fix")):
        return "attempt a code edit"

    if any(token in lowered for token in ("test", "pytest", "run ", "python ", "make ")):
        return "run or validate a hypothesis"

    words = normalized.split()
    if len(words) <= 5:
        return normalized
    return " ".join(words[:5]) + "..."


def summarize_chunks_for_panel(case_id: str, chunk_snippets: list[str]) -> list[str]:
    manual = MANUAL_REASONING_SUMMARIES.get(case_id)
    if manual and len(manual) == len(chunk_snippets):
        return list(manual)

    summaries: list[str] = []
    previous_text: str | None = None
    for index, text in enumerate(chunk_snippets):
        summaries.append(_summarize_generic_chunk(text, index, previous_text))
        previous_text = text
    return summaries


def _expand_limits_to_annotations(ax, annotations, pad_px: float = 6.0) -> None:
    if not annotations:
        return

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    annotation_bbox = Bbox.union([
        annotation.get_window_extent(renderer=renderer) for annotation in annotations
    ]).padded(pad_px)

    (label_x0, label_y0), (label_x1, label_y1) = ax.transData.inverted().transform([
        (annotation_bbox.x0, annotation_bbox.y0),
        (annotation_bbox.x1, annotation_bbox.y1),
    ])

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, label_x0), max(x1, label_x1))
    ax.set_ylim(min(y0, label_y0), max(y1, label_y1))


def render_panel_a(ax) -> None:
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    chunks = [
        (0.0, 7.0, "Step 1: rephrase\nthe problem"),
        (0.0, 4.8, "Step 2: identify\nrelevant principles"),
        (0.0, 2.6, "Step 3: derive\nan equation"),
        (0.0, 0.4, "Step 4: plug in\nnumbers"),
    ]
    chunk_y_centers = []
    for x, y, txt in chunks:
        box = FancyBboxPatch((x, y), 5.2, 1.6, boxstyle="round,pad=0.2",
                             linewidth=1.2, edgecolor=NEUTRAL, facecolor="#ebf4ff")
        ax.add_patch(box)
        ax.text(x + 2.6, y + 0.8, txt, ha="center", va="center", fontsize=11, color="black")
        chunk_y_centers.append(y + 0.8)

    axis_x = 8.8
    ax.plot([axis_x, axis_x], [0.5, 8.5], color=NEUTRAL, lw=2.5)
    ax.annotate("", xy=(axis_x, 9.2), xytext=(axis_x, 8.4),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=2.5, mutation_scale=20))
    ax.text(axis_x, 9.6, "macro", fontsize=14, color=NEUTRAL, ha="center")
    ax.text(axis_x, -0.2, "micro", fontsize=14, color=NEUTRAL, ha="center")
    
    # Placed on the right side so it doesn't overlap arrows
    ax.text(axis_x + 0.6, 4.5, "SLoD axis", fontsize=12, color=NEUTRAL,
            rotation=90, ha="center", va="center")

    slod_coords = [7.8, 5.6, 3.4, 1.2]
    for cy, sy in zip(chunk_y_centers, slod_coords):
        ax.annotate("", xy=(axis_x - 0.2, sy), xytext=(5.6, cy),
                    arrowprops=dict(arrowstyle="-|>", color=LIGHT, lw=2.0, mutation_scale=15))
        ax.scatter([axis_x], [sy], s=120, color=CORRECT, zorder=3,
                   edgecolor="white", linewidth=1.5)

    ax.text(0.0, 9.2, "Each chunk → one SLoD coordinate",
            fontsize=10, color=NEUTRAL, style="italic")


def render_panel_b(ax, bt_scores: list[float], chunk_labels: list[str],
                   caption: str) -> None:
    n = len(bt_scores)
    xs = np.arange(n)
    colors = [MACRO_C if r > 0 else MICRO_C for r in bt_scores]

    ax.axhline(0, color=NEUTRAL, lw=0.6, ls="--", alpha=0.6)
    for i in range(1, n):
        if (bt_scores[i] > 0) != (bt_scores[i - 1] > 0):
            ax.plot([i - 1, i], [bt_scores[i - 1], bt_scores[i]],
                    color=FLIP_C, lw=3.0, alpha=0.6, zorder=2)
    ax.plot(xs, bt_scores, color="k", lw=0.8, alpha=0.35, zorder=1)
    ax.scatter(xs, bt_scores, c=colors, s=180, zorder=4,
               edgecolor="white", linewidth=1.0)

    span = max(abs(min(bt_scores)), abs(max(bt_scores))) or 1.0
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(min(bt_scores) - span * 0.3, max(bt_scores) + span * 0.3)

    annotations = []
    for x, y, label in zip(xs, bt_scores, chunk_labels):
        snippet = label if len(label) <= 34 else label[:33].rstrip() + "…"
        yoff = -span * 0.18 if y < span * 0.3 else span * 0.14
        va = "top" if yoff < 0 else "bottom"
        if yoff < 0:
            if x == 0:
                xoff = 0.14
                ha = "left"
            elif x == n - 1:
                xoff = -0.14
                ha = "right"
            else:
                direction = -1 if x < 0.5 * (n - 1) else 1
                xoff = 0.22 * direction
                ha = "right" if direction < 0 else "left"
        else:
            xoff = 0.0
            ha = "center"

        annotations.append(ax.annotate(
            snippet, xy=(x, y), xytext=(x + xoff, y + yoff),
            fontsize=9, ha=ha, va=va,
            color=NEUTRAL, family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=LIGHT, linewidth=0.5),
        ))

    _expand_limits_to_annotations(ax, annotations)
    ax.set_xlabel("Reasoning chunk index", fontsize=9.5)
    ax.set_xticks(xs)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, alpha=0.18, axis="y")

    leg_macro = mpatches.Patch(color=MACRO_C, label="+ macro")
    leg_micro = mpatches.Patch(color=MICRO_C, label="− micro")
    leg_flip = mpatches.Patch(color=FLIP_C, alpha=0.6, label="sign-flip (thrashing)")
    ax.legend(handles=[leg_macro, leg_micro, leg_flip], loc="upper right",
              fontsize=9, frameon=True, framealpha=0.95)

    if caption:
        ax.text(0.5 * (n - 1), min(bt_scores) - span * 0.78, caption,
                ha="center", fontsize=10, color=NEUTRAL, style="italic")


def render_panel_c(ax) -> None:
    """Pass@1 reranking lift on FrontierScience Olympiad."""
    labels = ["Random\n(per-attempt avg)", "LightGBM\nSLoD scorer", "Pass@5\noracle"]
    values = [58.2, 68.0, 83.0]
    colors = ["#a0aec0", CORRECT, "#2d3748"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.75)
    
    # Hide top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color(NEUTRAL)
    ax.spines["bottom"].set_color(NEUTRAL)
    ax.tick_params(axis="both", colors=NEUTRAL, labelsize=10)

    for bar, v, c in zip(bars, values, colors):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold",
                color=c)

    ax.set_ylim(0, 100)
    ax.set_ylabel("Pass@1 (%)", fontsize=12, color=NEUTRAL)
    
    # Orthogonal arrow for +9.8 pp
    # Starting vertical line at y=64 to avoid overlapping 58.2% text
    ax.plot([0.0, 0.0], [64, 76.5], color=CORRECT, lw=2.0)
    ax.annotate("", xy=(1.0, 76.5), xytext=(0.0, 76.5),
                arrowprops=dict(arrowstyle="-|>", color=CORRECT, lw=2.0, mutation_scale=15))
    ax.text(0.5, 78.5, "+9.8 pp", ha="center", fontsize=12,
            color=CORRECT, fontweight="bold")
    
    ax.text(1.5, 95, "39.5% of oracle gap recovered",
            ha="center", fontsize=10, color=NEUTRAL, style="italic")


def save_panel_pdf(out_path: Path, render, *, figsize: tuple[float, float],
                   adjust: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    render(ax)
    fig.subplots_adjust(**adjust)
    fig.savefig(out_path, facecolor="white", bbox_inches="tight")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("SLoD-ICML-2026-Workshop/figures/fig_teaser_real.png"),
                        help="output PNG path")
    parser.add_argument("--rankings", type=Path, default=None,
                        help="optional chunk_rankings.jsonl to source Panel B")
    parser.add_argument("--id", type=str, default=DEFAULT_CASE_ID,
                        help="case id to extract from --rankings")
    parser.add_argument("--caption", type=str, default=DEFAULT_CASE_TITLE,
                        help="italic caption under Panel B")
    args = parser.parse_args()

    if args.rankings is not None:
        bt_scores, chunk_snippets = load_case_from_rankings(args.rankings, args.id)
    else:
        bt_scores = list(DEFAULT_BT_SCORES)
        chunk_snippets = list(DEFAULT_CHUNK_SNIPPETS)
    chunk_labels = summarize_chunks_for_panel(args.id, chunk_snippets)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    render_panel_a(axes[0])
    render_panel_b(axes[1], bt_scores, chunk_labels, args.caption)
    render_panel_c(axes[2])

    fig.subplots_adjust(left=0.035, right=0.99, top=0.85, bottom=0.18, wspace=0.42)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, facecolor="white")
    plt.close(fig)

    stem = args.out.stem
    panel_outputs = {
        "a": args.out.with_name(f"{stem}_a.pdf"),
        "b": args.out.with_name(f"{stem}_b.pdf"),
        "c": args.out.with_name(f"{stem}_c.pdf"),
    }
    save_panel_pdf(
        panel_outputs["a"],
        render_panel_a,
        figsize=(4.1, 3.1),
        adjust={"left": 0.03, "right": 0.98, "top": 0.98, "bottom": 0.08},
    )
    save_panel_pdf(
        panel_outputs["b"],
        lambda ax: render_panel_b(ax, bt_scores, chunk_labels, args.caption),
        figsize=(4.1*1.3, 3.1*1.3),
        adjust={"left": 0.12, "right": 0.98, "top": 0.98, "bottom": 0.16},
    )
    save_panel_pdf(
        panel_outputs["c"],
        render_panel_c,
        figsize=(4.1, 3.2),
        adjust={"left": 0.14, "right": 0.98, "top": 0.98, "bottom": 0.17},
    )

    save_panel_pdf(
        args.out.with_name(f"{stem}_b.png"),
        lambda ax: render_panel_b(ax, bt_scores, chunk_labels, args.caption),
        figsize=(4.1*1.3, 3.1*1.3),
        adjust={"left": 0.12, "right": 0.98, "top": 0.98, "bottom": 0.16},
    )

    print(f"saved {args.out}")
    for key in ("a", "b", "c"):
        print(f"saved {panel_outputs[key]}")


if __name__ == "__main__":
    main()
