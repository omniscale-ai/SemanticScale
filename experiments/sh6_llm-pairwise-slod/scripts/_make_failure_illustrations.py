"""One-off: render the SH6 failure-mode illustration figure.

FrontierScience cases come strictly from the OLYMPIAD subset
(``has_final_answer=True``), where ``rambling_overlong`` is the only detector
with a ``confirmed`` verdict in the by-origin/olympiad report. Each case shows
the side (reasoning or answer) where the relevant detector fires; long
rambling traces are rendered without per-chunk snippets because the visual
story is the length/sign-flip density itself.
"""

from __future__ import annotations

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

FS_RANK = (
    "/home/kna/SemanticScale/data/sh6/frontierscience/"
    "deepseek/deepseek-v3.2_reasoning-auto/chunk_rankings.jsonl"
)
SWE_RANK = (
    "/home/kna/SemanticScale/data/sh6/swe-agent-trajectories/"
    "model-all/chunk_rankings.jsonl"
)

CASES = [
    {
        "src": FS_RANK,
        "id": "84bba88d-2aee-4fcb-baab-4c6eea17b0d8",
        "side": "reasoning",
        "show_snippets": True,
        "title": (
            "FrontierScience · olympiad · 84bba88d · physics (ratchet) · CORRECT (positive control)\n"
            "no detector fires — short, almost-monotonic reasoning is the olympiad-success template"
        ),
    },
    {
        "src": FS_RANK,
        "id": "5c97845c-c039-48d8-abcf-703afc476b98",
        "side": "reasoning",
        "show_snippets": False,
        "title": (
            "FrontierScience · olympiad · 5c97845c · physics (popsicle-stick stability) · WRONG\n"
            "flags: rambling_overlong (reasoning length far above olympiad-success 90th pctile) + "
            "derailment_late + answer_drift"
        ),
    },
    {
        "src": FS_RANK,
        "id": "f10254b9-f0a1-407f-8af9-3169e23eee5e",
        "side": "reasoning",
        "show_snippets": False,
        "title": (
            "FrontierScience · olympiad · f10254b9 · physics (cylinder in stratified fluid) · WRONG\n"
            "flags: rambling_overlong + answer_meandering + answer_volatility"
        ),
    },
    {
        "src": FS_RANK,
        "id": "69bd11e8-5f96-45dc-8123-b9d800101430",
        "side": "reasoning",
        "show_snippets": True,
        "title": (
            "FrontierScience · olympiad · 69bd11e8 · physics (1-D domain wall) · WRONG, but no flags fire (false negative)\n"
            "21 sign flips in 31 chunks — same shape signature as failure, but trace too short to trip rambling_overlong"
        ),
    },
    {
        "src": SWE_RANK,
        "id": "AnalogJ__lexicon-336",
        "side": "reasoning",
        "show_snippets": True,
        "title": (
            "SWE-agent · AnalogJ__lexicon-336 · llama-70B · WRONG, exit_context\n"
            "flags: thrashing + truncation_abort  (reasoning-side detectors are pre-registered → unbiased)"
        ),
    },
    {
        "src": SWE_RANK,
        "id": "ReproNim__reproman-518",
        "side": "reasoning",
        "show_snippets": True,
        "title": (
            "SWE-agent · ReproNim__reproman-518 · llama-70B · WRONG\n"
            "flags: thrashing + rambling_overlong + no_commitment"
        ),
    },
]


def load_case(case: dict) -> None:
    with open(case["src"]) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("id") == case["id"]:
                case["rp"] = d.get(f"{case['side']}_params") or []
                case["rc"] = d.get(f"{case['side']}_chunks") or []
                return
    raise RuntimeError(f"id {case['id']} not found in {case['src']}")


def truncate(text: str, n: int) -> str:
    s = " ".join(text.replace("\n", " ").split())
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def render_panel(ax, case: dict) -> None:
    rp = case["rp"]
    rc = case["rc"]
    n = len(rp)
    x = np.arange(n)
    colors = ["#1f77b4" if r > 0 else "#d62728" for r in rp]

    ax.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.7)

    # Sign-flip edges first so they sit under the markers
    for i in range(1, n):
        if (rp[i] > 0) != (rp[i - 1] > 0):
            ax.plot(
                [i - 1, i], [rp[i - 1], rp[i]],
                color="orange", lw=2.5, alpha=0.55, zorder=2,
            )
    ax.plot(x, rp, color="k", lw=0.8, alpha=0.35, zorder=1)
    marker_size = 110 if n <= 35 else max(18, 600 / n)
    ax.scatter(x, rp, c=colors, s=marker_size, zorder=4, edgecolor="white", linewidth=0.8)

    ymin, ymax = min(rp), max(rp)
    span = max(abs(ymin), abs(ymax))
    pad_top = max(span * 0.30, 1.0)

    if case.get("show_snippets", True):
        # Reserve a bottom strip for staggered chunk-text annotations
        if n <= 8:
            snippet_len, n_rows, rot, fs = 80, 2, 12, 8.0
        elif n <= 14:
            snippet_len, n_rows, rot, fs = 55, 3, 18, 8.0
        elif n <= 22:
            snippet_len, n_rows, rot, fs = 36, 4, 28, 7.6
        else:
            snippet_len, n_rows, rot, fs = 22, 5, 38, 7.4
        label_strip = max(span * 2.0, 5.0)
        ax.set_ylim(ymin - label_strip, ymax + pad_top)
        row_step = label_strip / (n_rows + 1.5)
        base_y = ymin - row_step
        for i, (xi, yi, ci) in enumerate(zip(x, rp, rc)):
            snippet = truncate(ci, snippet_len)
            if not snippet:
                continue
            row = i % n_rows
            ytext = base_y - row * row_step
            ax.annotate(
                "", xy=(xi, yi), xytext=(xi, ytext),
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.4, alpha=0.5),
                zorder=2.5,
            )
            ax.text(xi, ytext, snippet, fontsize=fs, ha="center", va="top",
                    rotation=rot, color="black")
    else:
        # No per-chunk text. Reserve only modest bottom padding and add a
        # callout summarising the trace shape.
        ax.set_ylim(ymin - span * 0.5, ymax + pad_top)
        # Callout: first chunk + a representative middle chunk + last chunk
        if rc:
            sample_idxs = [0, n // 2, n - 1]
            for idx in sample_idxs:
                snippet = truncate(rc[idx], 65)
                ax.annotate(
                    snippet,
                    xy=(idx, rp[idx]),
                    xytext=(idx, ymin - span * 0.35),
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.5, alpha=0.5),
                    fontsize=8.0, ha="center", va="top", rotation=15, color="black",
                )

    n_flips = sum(1 for a, b in zip(rp, rp[1:]) if (a > 0) != (b > 0))
    flip_pct = 100 * n_flips / max(1, n - 1)
    ax.set_title(
        case["title"]
        + f"\n(n={n} {case['side']} chunks · range {min(rp):+.1f} … {max(rp):+.1f}"
        f" · {n_flips} sign flips ≈ {flip_pct:.0f}% of edges)",
        fontsize=11, loc="left",
    )
    ax.set_xlabel(f"{case['side']} chunk index")
    ax.set_ylabel("SLoD (BT score)\n+ macro / − micro")
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(True, alpha=0.2, axis="y")


def render(cases: list[dict], out_path: str) -> None:
    fig, axs = plt.subplots(len(cases), 1, figsize=(17.0, 4.6 * len(cases)))
    if len(cases) == 1:
        axs = [axs]
    for ax, case in zip(axs, cases):
        render_panel(ax, case)

    macro = mpatches.Patch(color="#1f77b4", label="+ macro / framing chunk")
    micro = mpatches.Patch(color="#d62728", label="− micro / detail chunk")
    flip = mpatches.Patch(color="orange", alpha=0.55, label="sign-flip edge (thrashing)")
    fig.legend(
        handles=[macro, micro, flip],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.997),
        fontsize=9,
        framealpha=0.95,
    )
    fig.suptitle(
        "SH6 SLoD failure-mode illustrations — "
        "FrontierScience (olympiad subset only) & SWE-agent",
        fontsize=13, y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.5)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print("saved", out_path)


def main() -> None:
    for c in CASES:
        load_case(c)
    out = (
        "/home/kna/SemanticScale/experiments/sh6_llm-pairwise-slod/"
        "reports/_cross_dataset/failure_examples_overlay.png"
    )
    render(CASES, out)


if __name__ == "__main__":
    main()
