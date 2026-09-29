import argparse
import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path


DIMENSIONS = ("SR", "CSC", "DQ", "CS", "QR", "IS")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    answers = read_jsonl(args.answers)
    if not answers:
        raise ValueError("No candidate answers were found")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    questions = read_jsonl(args.questions)
    expected_labels = {model["label"] for model in config["candidate_models"] if model.get("enabled", False)}
    expected_questions = {question["id"] for question in questions}
    if len(expected_questions) != len(questions) or not expected_labels:
        raise ValueError("The benchmark configuration needs unique questions and enabled candidates")
    runs = {row["run_label"] for row in answers}
    if runs != {config["run_label"]}:
        raise ValueError("Answers must belong to the configured benchmark run")
    seen = set()
    blind = []
    key = []
    for row in answers:
        pair = (row["candidate_label"], row["question_id"])
        if pair in seen:
            raise ValueError(f"Duplicate answer: {pair}")
        seen.add(pair)
        if not row.get("answer", "").strip() or row.get("finish_reason") == "length":
            raise ValueError(f"Empty or truncated answer: {pair}")
        source = json.dumps([row["run_label"], *pair, args.seed], ensure_ascii=False)
        response_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        blind.append({
            "response_id": response_id,
            "question_id": row["question_id"],
            "question": row["question"],
            "retrieved_passages": json.dumps(row.get("retrieved_passages", []), ensure_ascii=False),
            "answer": row["answer"],
        })
        key.append({
            "response_id": response_id,
            "run_label": row["run_label"],
            "candidate_label": row["candidate_label"],
            "question_id": row["question_id"],
            "condition": row.get("condition", ""),
        })
    expected_pairs = {(label, question_id) for label in expected_labels for question_id in expected_questions}
    if seen != expected_pairs:
        missing = sorted(expected_pairs - seen)
        unexpected = sorted(seen - expected_pairs)
        raise ValueError(f"Incomplete benchmark answer matrix: missing={missing}, unexpected={unexpected}")
    rng = random.Random(args.seed)
    rng.shuffle(blind)
    write_csv(args.blind, ["response_id", "question_id", "question", "retrieved_passages", "answer"], blind)
    write_csv(args.key, ["response_id", "run_label", "candidate_label", "question_id", "condition"], key)
    template = [
        {"response_id": row["response_id"], "evaluator_id": evaluator, **{dimension: "" for dimension in DIMENSIONS}}
        for row in blind for evaluator in ("E1", "E2", "E3")
    ]
    write_csv(args.score_template, ["response_id", "evaluator_id", *DIMENSIONS], template)
    print(f"Prepared {len(blind)} blinded responses and {len(template)} scoring rows")


def mean(values):
    return sum(values) / len(values)


def bootstrap_ci(values, seed, repetitions=10000):
    rng = random.Random(seed)
    samples = [mean([rng.choice(values) for _ in values]) for _ in range(repetitions)]
    samples.sort()
    return samples[int(0.025 * repetitions)], samples[int(0.975 * repetitions)]


def sign_flip_p(values):
    observed = abs(mean(values))
    if len(values) > 20:
        raise ValueError("Exact sign-flip test supports at most 20 paired questions")
    null = [abs(mean([sign * value for sign, value in zip(signs, values)]))
            for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def analyze(args):
    key_rows = read_csv(args.key)
    score_rows = read_csv(args.scores)
    key = {row["response_id"]: row for row in key_rows}
    if len(key) != len(key_rows):
        raise ValueError("Duplicate response IDs in the blind key")
    scores = defaultdict(dict)
    for row in score_rows:
        response_id = row["response_id"]
        evaluator_id = row["evaluator_id"].strip()
        if response_id not in key or evaluator_id not in {"E1", "E2", "E3"}:
            raise ValueError(f"Unknown response or evaluator: {response_id}, {evaluator_id}")
        if evaluator_id in scores[response_id]:
            raise ValueError(f"Duplicate score: {response_id}, {evaluator_id}")
        values = {dimension: float(row[dimension]) for dimension in DIMENSIONS}
        if any(not 1 <= value <= 5 for value in values.values()):
            raise ValueError(f"Score outside 1–5 range: {response_id}, {evaluator_id}")
        scores[response_id][evaluator_id] = values
    if set(scores) != set(key) or any(set(value) != {"E1", "E2", "E3"} for value in scores.values()):
        raise ValueError("The three-expert scoring matrix is incomplete")
    by_candidate = defaultdict(dict)
    detail = []
    for response_id, meta in key.items():
        dimensions = {name: mean([values[name] for values in scores[response_id].values()]) for name in DIMENSIONS}
        total = sum(dimensions.values())
        candidate = meta["candidate_label"]
        question_id = meta["question_id"]
        if question_id in by_candidate[candidate]:
            raise ValueError(f"Duplicate candidate/question pair: {candidate}, {question_id}")
        by_candidate[candidate][question_id] = total
        detail.append({**meta, **dimensions, "Total_30": total})
    question_sets = {candidate: set(values) for candidate, values in by_candidate.items()}
    if len({frozenset(values) for values in question_sets.values()}) != 1:
        raise ValueError("Candidate models were not scored on the same questions")
    if args.reference not in by_candidate:
        raise ValueError(f"Reference candidate not found: {args.reference}")
    summary = []
    for candidate, values in by_candidate.items():
        subset = [row for row in detail if row["candidate_label"] == candidate]
        summary.append({"candidate_label": candidate, "n_questions": len(values),
                        **{name: mean([row[name] for row in subset]) for name in DIMENSIONS},
                        "Total_30": mean(list(values.values()))})
    summary.sort(key=lambda row: row["Total_30"], reverse=True)
    paired = []
    reference_values = by_candidate[args.reference]
    for candidate, values in by_candidate.items():
        if candidate == args.reference:
            continue
        differences = [reference_values[q] - values[q] for q in sorted(reference_values)]
        low, high = bootstrap_ci(differences, args.seed)
        paired.append({"reference": args.reference, "comparator": candidate,
                       "n_questions": len(differences), "mean_difference_30": mean(differences),
                       "bootstrap_95_low": low, "bootstrap_95_high": high,
                       "exact_sign_flip_p": sign_flip_p(differences)})
    write_csv(args.out_dir / "expert_scores_by_response.csv", ["response_id", "run_label", "candidate_label", "question_id", "condition", *DIMENSIONS, "Total_30"], detail)
    write_csv(args.out_dir / "expert_model_summary.csv", ["candidate_label", "n_questions", *DIMENSIONS, "Total_30"], summary)
    write_csv(args.out_dir / "paired_comparisons.csv", ["reference", "comparator", "n_questions", "mean_difference_30", "bootstrap_95_low", "bootstrap_95_high", "exact_sign_flip_p"], paired)
    for row in summary:
        print(f"{row['candidate_label']}: {row['Total_30']:.3f}/30 across {row['n_questions']} questions")


def main():
    parser = argparse.ArgumentParser(description="Prepare blinded expert scoring or analyze complete expert scores.")
    parser.add_argument("command", choices=("prepare", "analyze"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("benchmark_config.json"))
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path(__file__).with_name("outputs") / "model_answers.jsonl")
    parser.add_argument("--blind", type=Path, default=Path(__file__).with_name("outputs") / "blind_responses.csv")
    parser.add_argument("--key", type=Path, default=Path(__file__).with_name("outputs") / "blind_key.csv")
    parser.add_argument("--score-template", type=Path, default=Path(__file__).with_name("outputs") / "expert_scores_template.csv")
    parser.add_argument("--scores", type=Path, default=Path(__file__).with_name("outputs") / "expert_scores.csv")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("outputs") / "summary")
    parser.add_argument("--reference", default="RAMAD-LoRA-prompt-RAG")
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else analyze(args)


if __name__ == "__main__":
    main()
