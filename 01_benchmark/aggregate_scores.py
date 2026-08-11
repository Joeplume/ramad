import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DIMENSIONS = ["SR", "CSC", "DQ", "CS", "QR", "IS"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def enabled_labels(config: dict, key: str):
    return [item["label"] for item in config.get(key, []) if item.get("enabled", False)]


def mean(values):
    if not values:
        raise ValueError("Cannot calculate a mean from an empty sequence")
    return sum(values) / len(values)


def normalized(values: dict):
    total = sum(values.values())
    if total == 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: value / total for key, value in values.items()}


def minmax(values: dict):
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {key: 1.0 / len(values) for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def iterative_weights(matrix, candidates, evaluators, max_iters, tolerance, epsilon, damping):
    anchors = [label for label in evaluators if label in candidates]
    uniform = {label: 1.0 / len(evaluators) for label in evaluators}
    if len(anchors) < 2:
        history = [{
            "k": 0,
            "delta_L1": 0.0,
            **{f"alpha_{label}": uniform[label] for label in evaluators},
        }]
        return uniform, history

    alpha = dict(uniform)
    history = []
    for iteration in range(1, max_iters + 1):
        model_scores = {
            candidate: sum(alpha[evaluator] * matrix[evaluator][candidate] for evaluator in evaluators)
            for candidate in candidates
        }
        anchor_scores = {anchor: model_scores[anchor] for anchor in anchors}
        transformed = {key: value + epsilon for key, value in minmax(anchor_scores).items()}
        transformed = normalized(transformed)
        expanded = {evaluator: transformed.get(evaluator, 0.0) for evaluator in evaluators}
        alpha_new = normalized({
            evaluator: (1.0 - damping) * expanded[evaluator] + damping * uniform[evaluator]
            for evaluator in evaluators
        })
        delta = sum(abs(alpha_new[label] - alpha[label]) for label in evaluators)
        row = {"k": iteration, "delta_L1": delta}
        row.update({f"alpha_{label}": alpha_new[label] for label in evaluators})
        row.update({f"Score_{label}": model_scores[label] for label in candidates})
        history.append(row)
        alpha = alpha_new
        if delta < tolerance:
            break
    return alpha, history


def write_csv(path: Path, fieldnames: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_rows(raw_rows, candidates, evaluators, selected_run_label=None):
    valid = []
    rejected = []
    seen = set()
    for row in raw_rows:
        if selected_run_label is not None and row.get("run_label", "") != selected_run_label:
            continue
        key = (row.get("run_label", ""), row.get("evaluator_label"), row.get("candidate_label"), row.get("question_id"))
        if key in seen:
            raise ValueError(f"Duplicate score record: {key}")
        seen.add(key)
        if row.get("parse_error"):
            rejected.append({**row, "aggregation_reason": "parse_error"})
            continue
        if row.get("candidate_label") not in candidates or row.get("evaluator_label") not in evaluators:
            rejected.append({**row, "aggregation_reason": "label_not_enabled_in_config"})
            continue
        scores = row.get("scores") or {}
        if any(dimension not in scores for dimension in DIMENSIONS):
            rejected.append({**row, "aggregation_reason": "missing_dimension"})
            continue
        for dimension in DIMENSIONS:
            value = float(scores[dimension])
            if not 1.0 <= value <= 5.0:
                raise ValueError(f"Score outside 1-5 range for {key}, {dimension}: {value}")
            valid.append({
                "run_label": row.get("run_label", ""),
                "evaluator": row["evaluator_label"],
                "candidate": row["candidate_label"],
                "question_id": row["question_id"],
                "dimension": dimension,
                "score": value,
            })
    if not valid:
        raise RuntimeError("No usable score rows were found")
    return valid, rejected


def aggregate(config_path: Path, scores_path: Path, output_dir: Path, max_iters, tolerance, epsilon, damping, run_label):
    config = load_json(config_path)
    candidates = enabled_labels(config, "candidate_models")
    evaluators = enabled_labels(config, "evaluator_models")
    if not candidates:
        raise RuntimeError("No candidate model is enabled in the selected configuration")
    if not evaluators:
        raise RuntimeError("No evaluator model is enabled in the selected configuration")

    raw_rows = load_jsonl(scores_path)
    available_run_labels = sorted({row.get("run_label", "") for row in raw_rows})
    if run_label is None and len(available_run_labels) > 1:
        raise RuntimeError(f"Multiple run labels are present; choose one with --run-label: {available_run_labels}")
    flat, rejected = prepare_rows(raw_rows, candidates, evaluators, run_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "all_scores_long.csv",
        ["run_label", "evaluator", "candidate", "question_id", "dimension", "score"],
        flat,
    )

    simple_values = defaultdict(list)
    for row in flat:
        simple_values[(row["candidate"], row["dimension"])].append(row["score"])
    simple_rows = []
    for candidate in candidates:
        row = {"Model": candidate}
        for dimension in DIMENSIONS:
            values = simple_values[(candidate, dimension)]
            if not values:
                raise RuntimeError(f"No scores found for candidate {candidate}, dimension {dimension}")
            row[dimension] = mean(values)
        row["Average"] = mean([row[dimension] for dimension in DIMENSIONS])
        row["Total_30"] = sum(row[dimension] for dimension in DIMENSIONS)
        simple_rows.append(row)
    simple_rows.sort(key=lambda row: row["Average"], reverse=True)
    score_columns = ["Model", *DIMENSIONS, "Average", "Total_30"]
    write_csv(output_dir / "simple_mean_scores.csv", score_columns, simple_rows)

    grouped = defaultdict(list)
    for row in flat:
        grouped[(row["dimension"], row["evaluator"], row["candidate"])].append(row["score"])
    missing = [
        (dimension, evaluator, candidate)
        for dimension in DIMENSIONS
        for evaluator in evaluators
        for candidate in candidates
        if not grouped[(dimension, evaluator, candidate)]
    ]
    if missing:
        preview = ", ".join("/".join(item) for item in missing[:12])
        raise RuntimeError(f"The evaluator-by-candidate score matrix is incomplete: {preview}")

    weighted_by_dimension = {candidate: {} for candidate in candidates}
    weights_by_dimension = {}
    for dimension in DIMENSIONS:
        matrix = {
            evaluator: {
                candidate: mean(grouped[(dimension, evaluator, candidate)])
                for candidate in candidates
            }
            for evaluator in evaluators
        }
        weights, history = iterative_weights(
            matrix,
            candidates,
            evaluators,
            max_iters=max_iters,
            tolerance=tolerance,
            epsilon=epsilon,
            damping=damping,
        )
        weights_by_dimension[dimension] = weights
        for candidate in candidates:
            weighted_by_dimension[candidate][dimension] = sum(
                weights[evaluator] * matrix[evaluator][candidate] for evaluator in evaluators
            )
        history_columns = [
            "k",
            "delta_L1",
            *[f"alpha_{label}" for label in evaluators],
            *[f"Score_{label}" for label in candidates],
        ]
        write_csv(output_dir / f"history_{dimension}.csv", history_columns, history)

    weighted_rows = []
    for candidate in candidates:
        row = {"Model": candidate, **weighted_by_dimension[candidate]}
        row["Average"] = mean([row[dimension] for dimension in DIMENSIONS])
        row["Total_30"] = sum(row[dimension] for dimension in DIMENSIONS)
        weighted_rows.append(row)
    weighted_rows.sort(key=lambda row: row["Average"], reverse=True)
    write_csv(output_dir / "weighted_final_scores.csv", score_columns, weighted_rows)
    write_csv(output_dir / "table_s4_updated.csv", score_columns, weighted_rows)

    weight_rows = []
    for dimension in DIMENSIONS:
        row = {"Dimension": dimension}
        row.update(weights_by_dimension[dimension])
        weight_rows.append(row)
    write_csv(output_dir / "evaluator_weights_by_dimension.csv", ["Dimension", *evaluators], weight_rows)

    if rejected:
        with (output_dir / "rejected_score_rows.jsonl").open("w", encoding="utf-8") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print("Weighted ranking")
    for rank, row in enumerate(weighted_rows, start=1):
        print(f"{rank}. {row['Model']}: Average={row['Average']:.3f}, Total_30={row['Total_30']:.3f}")
    print(f"Output directory: {output_dir}")
    if rejected:
        print(f"Excluded score records: {len(rejected)}")


def resolve(root: Path, value: str):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate six-dimensional Comment 8 scores with iterative evaluator weighting."
    )
    parser.add_argument("--config", default="benchmark_config.json")
    parser.add_argument("--scores", default="outputs/model_scores.jsonl")
    parser.add_argument("--out-dir", default="outputs/summary")
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--run-label", default=None)
    args = parser.parse_args()

    if args.max_iters < 1:
        parser.error("--max-iters must be at least 1")
    if not 0.0 <= args.damping <= 1.0:
        parser.error("--damping must be between 0 and 1")
    if args.tolerance <= 0 or args.epsilon <= 0:
        parser.error("--tolerance and --epsilon must be positive")

    root = Path(__file__).resolve().parent
    aggregate(
        resolve(root, args.config),
        resolve(root, args.scores),
        resolve(root, args.out_dir),
        args.max_iters,
        args.tolerance,
        args.epsilon,
        args.damping,
        args.run_label,
    )


if __name__ == "__main__":
    main()
