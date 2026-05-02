"""Render the 3-panel graphical-abstract teaser used in the paper.

Panels:
  (A) Schematic: a four-step reasoning trace projecting onto the SLoD axis.
  (B) Concrete real failure case: SWE-agent ReproNim/reproman-518, llama-70B.
      Loaded from chunk_rankings.jsonl when available; otherwise the hardcoded
      values below (extracted from the HTML illustration on origin/main) are used.
  (C) Pass@1 lift bar chart on the FrontierScience Olympiad subset.

Usage:
    python experiments/sh6_llm-pairwise-slod/scripts/_make_teaser_figure.py \
        [--out figures/fig_teaser_real.png] \
        [--rankings path/to/chunk_rankings.jsonl] [--id <case-id>]

Default output path lives next to the failure-mode illustration overlay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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
DEFAULT_CASE_TITLE = (
    "SWE-agent · ReproNim/reproman-518 · WRONG: 4 sign flips in 5 chunks → "
    r"$\mathtt{thrashing}$ + $\mathtt{no\_commitment}$ detectors fire"
)


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


def render_panel_a(ax) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("(A) From reasoning trace to SLoD trajectory",
                 fontsize=10.5, loc="left", pad=8)

    chunks = [
        (0.4, 7.3, "Step 1: rephrase\nthe problem"),
        (0.4, 5.2, "Step 2: identify\nrelevant principles"),
        (0.4, 3.1, "Step 3: derive\nan equation"),
        (0.4, 1.0, "Step 4: plug in\nnumbers"),
    ]
    chunk_y_centers = []
    for x, y, txt in chunks:
        box = FancyBboxPatch((x, y), 3.0, 1.6, boxstyle="round,pad=0.04",
                             linewidth=0.8, edgecolor=NEUTRAL, facecolor="#ebf4ff")
        ax.add_patch(box)
        ax.text(x + 1.5, y + 0.8, txt, ha="center", va="center", fontsize=8)
        chunk_y_centers.append(y + 0.8)

    axis_x = 7.8
    ax.plot([axis_x, axis_x], [0.5, 9.0], color=NEUTRAL, lw=1.5)
    ax.annotate("", xy=(axis_x, 9.4), xytext=(axis_x, 8.7),
                arrowprops=dict(arrowstyle="-|>", color=NEUTRAL, lw=1.5))
    ax.text(axis_x + 0.25, 9.3, "macro", fontsize=9, color=NEUTRAL)
    ax.text(axis_x + 0.25, 0.5, "micro", fontsize=9, color=NEUTRAL)
    ax.text(axis_x - 0.5, 4.75, "SLoD axis", fontsize=9, color=NEUTRAL,
            rotation=90, ha="center", va="center")

    slod_coords = [8.4, 6.2, 3.4, 1.4]
    for cy, sy in zip(chunk_y_centers, slod_coords):
        ax.annotate("", xy=(axis_x - 0.05, sy), xytext=(3.5, cy),
                    arrowprops=dict(arrowstyle="->", color=LIGHT, lw=0.9))
        ax.scatter([axis_x], [sy], s=45, color=CORRECT, zorder=3,
                   edgecolor="white", linewidth=0.8)

    ax.text(5.2, 9.7, "Each chunk → one SLoD coordinate",
            fontsize=8.5, color=NEUTRAL, style="italic")


def render_panel_b(ax, bt_scores: list[float], chunk_snippets: list[str],
                   caption: str) -> None:
    ax.set_title("(B) Real failure: agent thrashes between macro/micro",
                 fontsize=10.5, loc="left", pad=8)

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
    for i, (x, y, txt) in enumerate(zip(xs, bt_scores, chunk_snippets)):
        # Truncate to fit
        snippet = txt if len(txt) <= 32 else txt[:31].rstrip() + "…"
        yoff = -span * 0.18 if y < span * 0.3 else span * 0.14
        va = "top" if yoff < 0 else "bottom"
        ax.annotate(snippet, xy=(x, y), xytext=(x, y + yoff),
                    fontsize=7.6, ha="center", va=va,
                    color=NEUTRAL, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=LIGHT, linewidth=0.5))

    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(min(bt_scores) - span * 0.85, max(bt_scores) + span * 0.45)
    ax.set_xlabel("Reasoning chunk index", fontsize=9.5)
    ax.set_ylabel("SLoD (BT score)\n+ macro / − micro", fontsize=9)
    ax.set_xticks(xs)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(True, alpha=0.18, axis="y")

    leg_macro = mpatches.Patch(color=MACRO_C, label="+ macro")
    leg_micro = mpatches.Patch(color=MICRO_C, label="− micro")
    leg_flip = mpatches.Patch(color=FLIP_C, alpha=0.6, label="sign-flip (thrashing)")
    ax.legend(handles=[leg_macro, leg_micro, leg_flip], loc="upper right",
              fontsize=7.6, frameon=True, framealpha=0.95)

    ax.text(0.5 * (n - 1), min(bt_scores) - span * 0.78, caption,
            ha="center", fontsize=7.8, color=NEUTRAL, style="italic")


def render_panel_c(ax) -> None:
    """Pass@1 reranking lift on FrontierScience Olympiad."""
    ax.set_title("(C) SLoD reranking lifts Pass@1",
                 fontsize=10.5, loc="left", pad=8)

    labels = ["Random\n(per-attempt avg)", "LightGBM\nSLoD scorer", "Pass@5\noracle"]
    values = [58.2, 68.0, 83.0]
    colors = [LIGHT, CORRECT, NEUTRAL]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold",
                color=NEUTRAL)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Pass@1 (%)", fontsize=9.5)
    ax.annotate("", xy=(1, 68), xytext=(0, 58.2),
                arrowprops=dict(arrowstyle="->", color=CORRECT, lw=1.5))
    ax.text(0.5, 63, "+9.8 pp", ha="center", fontsize=9,
            color=CORRECT, fontweight="bold")
    ax.text(1.5, 90, "39.5% of oracle gap recovered",
            ha="center", fontsize=8.5, color=NEUTRAL, style="italic")
    ax.tick_params(axis="x", labelsize=8.5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("figures/fig_teaser_real.png"),
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

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 0.95]})
    render_panel_a(axes[0])
    render_panel_b(axes[1], bt_scores, chunk_snippets, args.caption)
    render_panel_c(axes[2])

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
