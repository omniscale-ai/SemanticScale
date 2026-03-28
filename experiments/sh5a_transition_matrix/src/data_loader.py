"""Stage A: Load and merge SH5 data into unified trace records."""

import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def group_steps_by_trace(steps: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group step records by (question_id, condition), sorted by step_index."""
    groups = defaultdict(list)
    for step in steps:
        key = (step["question_id"], step["condition"])
        groups[key].append(step)
    # Sort each group by step_index
    for key in groups:
        groups[key].sort(key=lambda s: s["step_index"])
    return dict(groups)


def build_traces(
    steps_by_trace: dict,
    answer_scores: list[dict],
    jump_metrics: list[dict],
) -> list[dict]:
    """Build unified trace records by merging steps, scores, and jump metrics."""
    # Index scores and jumps by (question_id, condition)
    score_map = {}
    for rec in answer_scores:
        key = (rec["question_id"], rec["condition"])
        score_map[key] = rec

    jump_map = {}
    for rec in jump_metrics:
        key = (rec["question_id"], rec["condition"])
        jump_map[key] = rec

    traces = []
    missing_scores = 0
    missing_jumps = 0

    for key, step_list in steps_by_trace.items():
        qid, condition = key

        # Extract hard labels and probability vectors
        hard_labels = [s["predicted_slod"] for s in step_list]
        prob_vectors = [
            [s["probabilities"]["macro"], s["probabilities"]["meso"], s["probabilities"]["micro"]]
            for s in step_list
        ]

        # Get scores
        score = score_map.get(key)
        if score is None:
            missing_scores += 1
            continue

        # Get jump metrics
        jump = jump_map.get(key)
        if jump is None:
            missing_jumps += 1
            jump_count = None
            normalized_jump_rate = None
        else:
            jump_count = jump["jump_count"]
            normalized_jump_rate = jump["normalized_jump_rate"]

        trace = {
            "question_id": qid,
            "condition": condition,
            "n_steps": len(step_list),
            "hard_labels": hard_labels,
            "prob_vectors": prob_vectors,
            "answer_token_f1": score["answer_token_f1"],
            "attribution_f1": score["attribution_f1"],
            "jump_count": jump_count,
            "normalized_jump_rate": normalized_jump_rate,
        }
        traces.append(trace)

    if missing_scores > 0:
        print(f"WARNING: {missing_scores} traces missing answer scores")
    if missing_jumps > 0:
        print(f"WARNING: {missing_jumps} traces missing jump metrics")

    return traces


def validate_traces(traces: list[dict]) -> None:
    """Run validation checks on traces."""
    assert len(traces) == 2000, f"Expected 2000 traces, got {len(traces)}"

    for t in traces:
        assert len(t["hard_labels"]) == t["n_steps"], (
            f"Label count mismatch for {t['question_id']}/{t['condition']}"
        )
        assert len(t["prob_vectors"]) == t["n_steps"]
        for pv in t["prob_vectors"]:
            total = sum(pv)
            assert abs(total - 1.0) < 0.01, f"Prob vector sums to {total}"

    # Distribution of n_steps
    from collections import Counter
    step_dist = Counter(t["n_steps"] for t in traces)
    print("Step distribution:")
    for k in sorted(step_dist):
        print(f"  {k} steps: {step_dist[k]} traces")
    print(f"Total traces: {len(traces)}")


def save_traces(traces: list[dict], output_path: str) -> None:
    """Save traces to JSONL."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print(f"Saved {len(traces)} traces to {output_path}")


def load_all(data_dir: str = "../../data/sh5", output_dir: str = "../../data/sh5a") -> list[dict]:
    """Main entry: load all data, merge, validate, save."""
    output_path = Path(output_dir) / "traces.jsonl"

    print("Loading SH5 data files...")
    steps = load_jsonl(f"{data_dir}/cot_slod_tags.jsonl")
    print(f"  cot_slod_tags: {len(steps)} steps")

    scores = load_jsonl(f"{data_dir}/answer_scores.jsonl")
    print(f"  answer_scores: {len(scores)} records")

    jumps = load_jsonl(f"{data_dir}/jump_metrics.jsonl")
    print(f"  jump_metrics: {len(jumps)} records")

    print("Grouping steps by trace...")
    steps_by_trace = group_steps_by_trace(steps)
    print(f"  {len(steps_by_trace)} traces found")

    print("Building unified traces...")
    traces = build_traces(steps_by_trace, scores, jumps)

    print("Validating...")
    validate_traces(traces)

    save_traces(traces, str(output_path))
    return traces
