"""Generate one self-contained interactive HTML per illustration case.

Each page shows the SLoD (BT score) trajectory at the top and the full
chunk-by-chunk text below. Clicking a marker in the chart scrolls to that
chunk; MathJax renders any LaTeX that's in the chunk text.
"""

from __future__ import annotations

import html as html_lib
import json
import os
from textwrap import dedent

FS_RANK = (
    "/home/kna/SemanticScale/data/sh6/frontierscience/"
    "deepseek/deepseek-v3.2_reasoning-auto/chunk_rankings.jsonl"
)
SWE_RANK = (
    "/home/kna/SemanticScale/data/sh6/swe-agent-trajectories/"
    "model-all/chunk_rankings.jsonl"
)
TRACE_FS = (
    "/home/kna/SemanticScale/data/sh6/frontierscience/"
    "deepseek/deepseek-v3.2_reasoning-auto/traces.jsonl"
)
TRACE_SWE = "/home/kna/SemanticScale/data/sh6/swe-agent-trajectories/model-all/traces.jsonl"

OUT_DIR = (
    "/home/kna/SemanticScale/experiments/sh6_llm-pairwise-slod/"
    "reports/_cross_dataset/illustrations"
)


CASES = [
    dict(
        slug="fs_olympiad_5c97845c_popsicle_stick",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="5c97845c-c039-48d8-abcf-703afc476b98",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="physics (popsicle-stick stability)",
        verdict="WRONG",
        flags="rambling_overlong + derailment_late + answer_drift",
        failure_summary=(
            "<b>Verified failure — wrong variable in the final scaling.</b> "
            "The problem asks for <code>H</code> in terms of length <code>L</code>, "
            "thickness <code>e</code>, density <code>ρ</code>, Young's modulus <code>E</code>, and <code>g</code>. "
            "Reference answer: <code>H ≈ (E/ρg)·(e/L)⁴</code>. "
            "Model's final answer: <code>H ∝ E·e⁴ / (ρ·g·w⁴)</code> — "
            "it puts the stick <i>width</i> <code>w</code> in the denominator instead of the <i>length</i> <code>L</code>. "
            "<code>w</code> is not even a permitted variable in the requested expression. "
            "The 189-chunk reasoning trace contains the right intermediate steps "
            "(elastic energy of bending, mass per stick, equating to gravitational PE), "
            "but the final dimensional substitution swaps <code>L</code> for <code>w</code> — "
            "the answer never recovers. The grader confirms this exact mismatch "
            "(reference uses <code>L</code>, student uses <code>w</code>)."
        ),
    ),
    dict(
        slug="fs_olympiad_f10254b9_magnus_cylinder",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="f10254b9-f0a1-407f-8af9-3169e23eee5e",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="physics (cylinder under Magnus force in stratified fluid)",
        verdict="WRONG",
        flags="rambling_overlong + answer_meandering + answer_volatility",
        failure_summary=(
            "<b>Verified failure — overall sign error on the trajectory.</b> "
            "Reference: <code>x(z) = −√(−z(2g/(k²ω²) + z)) + (g/(k²ω²))·arccos(1 + zk²ω²/g)</code>. "
            "Model: <code>x(z) = (g/(k²ω²))·[ √(1 − (1+(k²ω²/g)z)²) − arccos(1+(k²ω²/g)z) ]</code>. "
            "The square-root argument is algebraically identical (the model's "
            "<code>1 − (1+u)²</code> factors to <code>−u(2+u)</code> after pulling out "
            "<code>(g/k²ω²)²</code>, which matches the reference's <code>−z(2g/k²ω² + z)</code>). "
            "But the model has <code>+√(…) − arccos(…)</code> while the reference has <code>−√(…) + arccos(…)</code> — "
            "the trajectory is the <i>negative</i> of the correct one. "
            "Cylinder rises instead of falls (or vice versa). "
            "The 181-chunk reasoning trace shows extensive bookkeeping with no clean place where the sign convention is fixed; "
            "the error is consistent with the agent never re-checking the boundary condition direction."
        ),
    ),
    dict(
        slug="fs_olympiad_f88cc0c0_figure_eight",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="f88cc0c0-2ca6-4cb0-bd1e-a37ddcf11294",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="physics (3-body figure-8 orbit, total energy)",
        verdict="WRONG",
        flags="rambling_overlong",
        failure_summary=(
            "<b>Verified failure — dropped the negative sign on a bound-state total energy.</b> "
            "Reference answer: <code>E = −8.6 × 10³⁹ J</code> (gravitationally bound system). "
            "Model's final answer: <code>8.6 × 10³⁹ J</code> — correct magnitude, wrong sign. "
            "Bound gravitational systems by definition have <code>E &lt; 0</code>; reporting a positive total energy "
            "implies the three stars are unbound, which contradicts the figure-8 orbit premise the model itself was reasoning about. "
            "In a 212-chunk reasoning trace the model does compute the kinetic and potential terms with the right magnitudes "
            "(KE at <code>O</code> = <code>(3/4)mv²</code>, PE at <code>O</code> = <code>−5Gm²/(2d)</code>), but loses track of the sign at the final aggregation."
        ),
    ),
    dict(
        slug="swe_AnalogJ_lexicon_336_wrong_file_detour",
        rank_src=SWE_RANK, trace_src=TRACE_SWE,
        item_id="AnalogJ__lexicon-336",
        side="reasoning",
        dataset_label="SWE-agent · llama-70B",
        subject="AnalogJ/lexicon issue 336 (string output → table formatter crashes)",
        verdict="WRONG",
        flags="thrashing + truncation_abort  (reasoning-side detectors are pre-registered → unbiased)",
        failure_summary=(
            "<b>Verified failure — superficial patch, plus a wrong-file edit detour.</b> "
            "The issue: <code>generate_table_result</code> in <code>lexicon/cli.py</code> assumes <code>output</code> is a list of records, "
            "but some providers return a bare string (a record id), causing a <code>TypeError</code>. "
            "The agent locates the function correctly (chunks 0–5), but its first patch (chunk 5) just wraps the string in a list — "
            "downstream code then iterates character-by-character and crashes with <code>'str' has no attribute 'get'</code>. "
            "On chunk 9 it pivots to a synthetic-dict workaround "
            "(<code>[{'id': output, 'type': '', 'name': '', 'content': '', 'ttl': ''}]</code>) — but applies the edit to <code>reproduce.py</code> "
            "(its own scratch reproducer) instead of <code>cli.py</code>. Chunk 10 realises the mistake and re-opens <code>cli.py</code>; "
            "chunk 12 hits an indentation error and re-edits. By chunk 14 the reproducer 'runs' but only because the synthetic-dict "
            "trick masks the underlying type error — the hidden test suite still rejects the patch. "
            "Failure: agent confused <i>which file is open</i>, then declared victory on a workaround that doesn't fix the real "
            "string-vs-record contract bug."
        ),
    ),
    dict(
        slug="swe_ReproNim_reproman_518_search_loop",
        rank_src=SWE_RANK, trace_src=TRACE_SWE,
        item_id="ReproNim__reproman-518",
        side="reasoning",
        dataset_label="SWE-agent · llama-70B",
        subject="ReproNim/reproman issue 518",
        verdict="WRONG, exit_context",
        flags="thrashing + rambling_overlong + no_commitment",
        failure_summary=(
            "<b>Verified failure — agent never located the relevant code.</b> "
            "All five chunks fight the search tool: "
            "(0) <code>search_dir \"run\" src</code> — fails because there's no <code>src/</code> dir; "
            "(1) <code>search_dir \"run\"</code> — bare search returns too much; "
            "(2) <code>search_dir \"run\" --name \"*.py\"</code> — invented flag, syntax error; "
            "(3) same invented flag, retried verbatim; "
            "(4) falls back to <code>grep -r \"run\" .</code>, which would dump the whole repo. "
            "Then the trace ends (<code>exit_context</code>). "
            "The agent never opened a file, never read the issue text, never made an edit. "
            "Failure mode: stuck at the very first navigation step, unable to settle on a search invocation that works — "
            "the textbook <code>thrashing</code> + <code>no_commitment</code> case the SLoD detectors are designed to catch."
        ),
    ),
    dict(
        slug="fs_olympiad_bdb3fc5f_relativistic_conductor",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="bdb3fc5f-9374-4e37-9af9-6036e3e29093",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="physics (relativistic moving conductor)",
        verdict="WRONG",
        flags="rambling_overlong + derailment_late + answer_drift",
        failure_summary=(
            "<b>The Error:</b> The problem asks for the induced current density $\\vec{J}(t)$ of a semi-infinite perfect conductor moving at relativistic velocity $v$. The model derived the first-order approximation: $\\vec{J}(t) \\propto (1 - v/c)\\cos(\\omega(1 - v/c)t)$. However, the exactly correct relativistic expression requires the full Lorentz transformations, yielding the second-order terms: $\\vec{J}(t) \\propto (1 - v^2/c^2)\\cos(\\omega(1 - v^2/c^2)t)$.<br><br>"
            "<b>Interpretation:</b> The model correctly reasoned through the boundary conditions for the majority of the trace but failed to stick to the exact Lorentz transformations, drifting into a non-relativistic or first-order approximation at the very end of the derivation. The SLoD trajectory clearly maps a late-stage derailment and drift as the model abandoned the exact calculus to simplify the final terms."
        ),
    ),
    dict(
        slug="fs_olympiad_f3ba1aae_oscillating_elliptical_object",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="f3ba1aae-2fc3-4d9b-a5a3-42bb91de4d7d",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="physics (oscillating elliptical object)",
        verdict="WRONG",
        flags="answer_volatility",
        failure_summary=(
            "<b>The Error:</b> The model was tasked with finding the frequency of oscillation $\\omega$ for a uniform 3D object with an elliptical cross-section. The model output $\\omega = 2\\sqrt{\\frac{g(A^{2}-B^{2})}{B(A^{2}+5B^{2})}}$. The correct answer is exactly the reciprocal of the terms inside the square root, and with a factor of $\\pi$ instead of $2$: $\\omega = \\pi\\sqrt{\\frac{(A^{2}+5B^{2})B}{g(A^{2}-B^{2})}}$.<br><br>"
            "<b>Interpretation:</b> The model inverted the relationship between the restoring torque and the moment of inertia. The <code>answer_volatility</code> SLoD flag perfectly captures this: the model frequently flipped the placement of its physical terms (numerator vs. denominator) and waffled between assumptions, leading to an inverted physical expression."
        ),
    ),
    dict(
        slug="fs_olympiad_8b695bb3_scrna_seq_tumor",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="8b695bb3-ea3c-4372-9222-9f0b79a68b6c",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="biology (scRNA-seq tumor biopsy)",
        verdict="WRONG",
        flags="answer_volatility + answer_uncommitted",
        failure_summary=(
            "<b>The Error:</b> The question had two blanks: it asked the model to identify a non-biological artifact separating samples (\"batch\" effects) and a specific method to visualize cell populations (\"UMAP\"). The model merely answered <code>t-SNE</code>. It entirely ignored the first question and provided an alternative (but incorrect compared to the reference) clustering algorithm for the second.<br><br>"
            "<b>Interpretation:</b> The model recognized that two answers were needed but failed to firmly commit to the \"batch effect\" diagnosis. This lack of commitment and volatility in the internal semantic representations flagged exactly the moment the model decided to silently drop the first half of the question and just throw out a single dimensionality reduction acronym."
        ),
    ),
    dict(
        slug="fs_olympiad_ecb58594_high_temp_bismuth",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="ecb58594-7fd9-4a94-b799-0f1c13acb752",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="chemistry (high-temperature bismuth reaction)",
        verdict="RIGHT",
        flags="none",
        failure_summary=(
            "<b>The Result:</b> Flawless deduction. The model confidently recognized the heavy element (Bismuth), the oxidation state changes with bromine/NaOH, and assembled the correct mass (>230). The trace was direct and lacked any meandering."
        ),
    ),
    dict(
        slug="fs_olympiad_4a737e9f_monovalent_anion",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="4a737e9f-5afd-4a14-a17b-c468cda1a8bb",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="chemistry (monovalent anion deduction)",
        verdict="RIGHT",
        flags="none",
        failure_summary=(
            "<b>The Result:</b> The model correctly identified Nitrogen as the versatile element forming multiple monovalent anions (like nitrite, nitrate, etc.) and swiftly constructed the target sodium salt without getting stuck or rambling."
        ),
    ),
    dict(
        slug="fs_olympiad_22f5a060_low_spin_diamagnetic",
        rank_src=FS_RANK, trace_src=TRACE_FS,
        item_id="22f5a060-b2ba-4d68-8ad1-cbcbd58a788e",
        side="reasoning",
        dataset_label="FrontierScience · olympiad",
        subject="chemistry (low-spin diamagnetic complex)",
        verdict="RIGHT",
        flags="none",
        failure_summary=(
            "<b>The Result:</b> This requires navigating coordination chemistry, mass percentages, and spin states simultaneously. The model methodically locked onto Cobalt(III) and Nitrite ligands, calculating the mass precisely, culminating in a highly confident prediction with no SLoD instability."
        ),
    ),
]


def load_jsonl(path: str, item_id: str) -> dict:
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("id") == item_id:
                return d
    raise RuntimeError(f"id {item_id} not found in {path}")


def truncate_preview(text: str, n: int = 140) -> str:
    s = " ".join(text.replace("\n", " ").split())
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


PAGE_TEMPLATE = dedent("""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SH6 illustration — __TITLE_SHORT__</title>
<script>
window.MathJax = {
  tex: { inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] },
  options: { renderActions: { addMenu: [] } }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         max-width: 1200px; margin: 1.2em auto; padding: 0 1em; color: #222; }
  h1 { font-size: 1.15em; margin-bottom: 0.1em; }
  .subtitle { font-size: 0.9em; color: #555; line-height: 1.45; margin-bottom: 1em; }
  .stats { font-family: monospace; font-size: 0.82em; color: #333;
           background: #f4f4f4; padding: 0.45em 0.7em; border-radius: 4px;
           display: inline-block; margin-top: 0.4em; }
  #plot { height: 460px; margin-bottom: 0.6em; }
  .legend { font-size: 0.82em; color: #666; margin: 0.4em 0 1em 0; }
  .legend code { background: #f0f0f0; padding: 0 0.3em; border-radius: 3px; }
  #chunks h3 { font-size: 0.95em; margin: 1.2em 0 0.5em 0; color: #555; font-weight: 600;
               border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
  .chunk { padding: 0.55em 0.8em; margin: 0.5em 0; border-radius: 6px;
           border-left: 4px solid #ccc; background: #fafafa; transition: background 0.15s; }
  .chunk.macro { border-left-color: #1f77b4; background: #f0f7ff; }
  .chunk.micro { border-left-color: #d62728; background: #fff3f0; }
  .chunk.highlight { background: #fff3a8 !important; box-shadow: 0 0 0 2px #f59f00; }
  .chunk-meta { font-size: 0.78em; color: #666; font-family: monospace; margin-bottom: 0.3em; }
  .chunk-meta .score { font-weight: 600; }
  .chunk-meta .score.pos { color: #1f77b4; }
  .chunk-meta .score.neg { color: #d62728; }
  .chunk-meta .flip { color: #d68000; font-weight: 600; }
  .chunk-text { white-space: pre-wrap; line-height: 1.45; font-size: 0.93em;
                word-wrap: break-word; overflow-wrap: break-word; }
  details.problem { margin: 0.4em 0 1em 0; }
  details.problem summary { cursor: pointer; font-size: 0.88em; color: #555;
                             padding: 0.3em 0.5em; background: #f4f4f4; border-radius: 4px; }
  details.problem .body { padding: 0.6em 0.8em; border: 1px solid #eee; border-top: 0; font-size: 0.9em; line-height: 1.45; white-space: pre-wrap; }
  .diagnosis { background: #fff8e0; border: 1px solid #f0d878; border-left: 4px solid #d6a000;
               padding: 0.7em 0.9em; margin: 0.6em 0 1em 0; line-height: 1.5; font-size: 0.92em; }
  .diagnosis code { background: rgba(0,0,0,0.06); padding: 0 0.25em; border-radius: 3px; font-size: 0.92em; }
  a.back { font-size: 0.85em; color: #1f77b4; text-decoration: none; }
  a.back:hover { text-decoration: underline; }
</style>
</head>
<body>
<a class="back" href="index.html">← back to index</a>
<h1>__TITLE_FULL__</h1>
<div class="subtitle">__SUBTITLE__</div>
<div class="stats">__STATS__</div>
<div class="diagnosis">__FAILURE_SUMMARY__</div>
__PROBLEM_BLOCK__
<div id="plot"></div>
<p class="legend">
  Click any point on the chart to jump to that chunk below.
  Blue = <code>+</code> macro/framing chunk · red = <code>−</code> micro/detail chunk ·
  orange edges = sign flip (the visual signature of <code>thrashing</code>).
</p>
<div id="chunks">
<h3>__SIDE_LABEL__ chunks (in order)</h3>
__CHUNK_HTML__
</div>
<script>
const trajectory = __TRAJ_JSON__;
const colors = trajectory.map(d => d.r > 0 ? '#1f77b4' : '#d62728');
const previews = trajectory.map(d => d.preview);

const trace = {
  x: trajectory.map((_, i) => i),
  y: trajectory.map(d => d.r),
  mode: 'lines+markers',
  marker: { color: colors, size: 11, line: { color: 'white', width: 1 } },
  line: { color: 'rgba(0,0,0,0.35)', width: 1 },
  text: previews,
  hovertemplate: '<b>chunk %{x}</b> · BT %{y:+.2f}<br>%{text}<br><span style="font-size:0.8em;color:#888">click to jump to text</span><extra></extra>'
};

const flipShapes = [];
for (let i = 1; i < trajectory.length; i++) {
  const a = trajectory[i-1].r, b = trajectory[i].r;
  if ((a > 0) !== (b > 0)) {
    flipShapes.push({
      type: 'line',
      x0: i-1, x1: i, y0: a, y1: b,
      line: { color: 'orange', width: 3.5 },
      layer: 'below'
    });
  }
}

const layout = {
  margin: { t: 20, l: 60, r: 20, b: 50 },
  xaxis: { title: 'chunk index', zeroline: false },
  yaxis: { title: 'SLoD (BT score) — +macro / −micro', zeroline: true, zerolinecolor: '#bbb' },
  shapes: flipShapes,
  hovermode: 'closest',
  hoverlabel: { align: 'left', bgcolor: 'white', font: { size: 12 } }
};

Plotly.newPlot('plot', [trace], layout, { displayModeBar: false, responsive: true });

document.getElementById('plot').on('plotly_click', e => {
  const idx = e.points[0].pointIndex;
  document.querySelectorAll('.chunk').forEach(el => el.classList.remove('highlight'));
  const target = document.getElementById('chunk-' + idx);
  if (target) {
    target.classList.add('highlight');
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
});
</script>
</body>
</html>
""")


def render_case(case: dict, index_entries: list) -> None:
    rank = load_jsonl(case["rank_src"], case["item_id"])
    trace = load_jsonl(case["trace_src"], case["item_id"])

    side = case["side"]
    rp = rank.get(f"{side}_params") or []
    rc = rank.get(f"{side}_chunks") or []
    n = len(rp)
    if n == 0:
        print(f"skip {case['slug']}: no {side} chunks")
        return

    n_flips = sum(1 for a, b in zip(rp, rp[1:]) if (a > 0) != (b > 0))
    flip_pct = 100 * n_flips / max(1, n - 1)
    rmin, rmax = min(rp), max(rp)

    title_short = f"{case['dataset_label']} · {case['item_id'][:14]}"
    title_full = (
        f"{case['dataset_label']} · {case['item_id']} · {case['subject']} · {case['verdict']}"
    )
    subtitle = f"<b>flags:</b> {html_lib.escape(case['flags'])}"
    stats = (
        f"n = {n} {side} chunks &nbsp;·&nbsp; "
        f"BT range {rmin:+.1f} … {rmax:+.1f} &nbsp;·&nbsp; "
        f"{n_flips} sign flips ({flip_pct:.0f}% of edges)"
    )

    problem_text = trace.get("problem") or ""
    if problem_text and len(problem_text.strip()) > 5:
        problem_block = (
            f'<details class="problem"><summary>📄 problem statement</summary>'
            f'<div class="body">{html_lib.escape(problem_text)}</div></details>'
        )
    else:
        problem_block = ""

    # Trajectory data for the chart (preview = HTML-escaped truncated text)
    traj = []
    for r, c in zip(rp, rc):
        traj.append({"r": r, "preview": html_lib.escape(truncate_preview(c, 140))})

    # Chunk list HTML — escape so HTML stays safe but LaTeX delimiters survive
    chunk_html_parts = []
    for i, (r, c) in enumerate(zip(rp, rc)):
        cls = "macro" if r > 0 else "micro"
        score_cls = "pos" if r > 0 else "neg"
        flip_marker = ""
        if i > 0 and ((rp[i] > 0) != (rp[i - 1] > 0)):
            flip_marker = ' <span class="flip">⟂ sign flip</span>'
        chunk_html_parts.append(
            f'<div class="chunk {cls}" id="chunk-{i}">'
            f'<div class="chunk-meta">'
            f'#{i:03d} · <span class="score {score_cls}">BT {r:+.2f}</span>'
            f'{flip_marker}'
            f'</div>'
            f'<div class="chunk-text">{html_lib.escape(c)}</div>'
            f'</div>'
        )
    chunk_html = "\n".join(chunk_html_parts)

    page = (PAGE_TEMPLATE
            .replace("__TITLE_SHORT__", html_lib.escape(title_short))
            .replace("__TITLE_FULL__", html_lib.escape(title_full))
            .replace("__SUBTITLE__", subtitle)
            .replace("__STATS__", stats)
            .replace("__FAILURE_SUMMARY__", case.get("failure_summary", ""))
            .replace("__PROBLEM_BLOCK__", problem_block)
            .replace("__SIDE_LABEL__", side)
            .replace("__CHUNK_HTML__", chunk_html)
            .replace("__TRAJ_JSON__", json.dumps(traj))
            )

    out_path = os.path.join(OUT_DIR, f"{case['slug']}.html")
    with open(out_path, "w") as f:
        f.write(page)
    print(f"wrote {out_path}  ({n} chunks, {n_flips} flips)")

    index_entries.append({
        "slug": case["slug"],
        "title": title_full,
        "flags": case["flags"],
        "verdict": case["verdict"],
        "n": n, "flips": n_flips, "flip_pct": flip_pct,
        "dataset": case["dataset_label"],
        "failure_summary": case.get("failure_summary", ""),
    })


def render_index(entries: list) -> None:
    rows = []
    for e in entries:
        rows.append(
            f'<li><a href="{e["slug"]}.html"><b>{html_lib.escape(e["title"])}</b></a><br>'
            f'<span class="meta">n={e["n"]} chunks · {e["flips"]} sign flips '
            f'({e["flip_pct"]:.0f}% of edges) · {html_lib.escape(e["verdict"])}</span><br>'
            f'<span class="flags">flags: {html_lib.escape(e["flags"])}</span>'
            f'<div class="diag">{e["failure_summary"]}</div>'
            f'</li>'
        )
    html = dedent(f"""\
    <!DOCTYPE html><html><head><meta charset="utf-8">
    <title>SH6 SLoD failure illustrations — index</title>
    <style>
      body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 2em auto; padding: 0 1em; color: #222; }}
      h1 {{ font-size: 1.3em; }}
      ul {{ list-style: none; padding: 0; }}
      li {{ padding: 0.7em 0.9em; margin: 0.5em 0; border: 1px solid #eee; border-radius: 6px; line-height: 1.5; }}
      li a {{ color: #1f77b4; text-decoration: none; font-size: 1.0em; }}
      li a:hover {{ text-decoration: underline; }}
      .meta {{ font-size: 0.85em; color: #555; font-family: monospace; }}
      .flags {{ font-size: 0.85em; color: #444; }}
      .diag {{ background: #fff8e0; border-left: 3px solid #d6a000; padding: 0.5em 0.7em;
               margin-top: 0.5em; font-size: 0.88em; line-height: 1.45; }}
      .diag code {{ background: rgba(0,0,0,0.06); padding: 0 0.25em; border-radius: 3px; }}
    </style></head><body>
    <h1>SH6 SLoD failure-mode illustrations (interactive)</h1>
    <p>Each page shows the SLoD (BT score) trajectory at the top and the full chunk text below.
    Click a point in the chart to jump to that chunk; sign-flip edges are highlighted in orange.
    LaTeX in the chunks is rendered by MathJax.</p>
    <ul>
    {os.linesep.join(rows)}
    </ul>
    </body></html>
    """)
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(html)
    print("wrote", os.path.join(OUT_DIR, "index.html"))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    entries = []
    for case in CASES:
        render_case(case, entries)
    render_index(entries)


if __name__ == "__main__":
    main()
