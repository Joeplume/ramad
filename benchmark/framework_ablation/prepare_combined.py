"""Merge the archived prompt-RAG condition with newly generated bare baselines."""

import argparse
import json
from pathlib import Path


RUN_LABEL = "frontier-framework-ablation-cn-20261001"
PROMPT_LABELS = {"Claude-Fable-5-prompt-RAG", "Kimi-K3-prompt-RAG"}
BARE_LABELS = {"Claude-Fable-5-bare", "Kimi-K3-bare"}


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize(rows, labels):
    selected = []
    for row in rows:
        if row.get("candidate_label") not in labels:
            continue
        item = dict(row)
        item["run_label"] = RUN_LABEL
        selected.append(item)
    return selected


def require_matrix(rows, labels, questions, evaluators=None):
    if evaluators is None:
        keys = {(row["candidate_label"], row["question_id"]) for row in rows}
        expected = {(label, question) for label in labels for question in questions}
    else:
        keys = {(row["evaluator_label"], row["candidate_label"], row["question_id"]) for row in rows}
        expected = {(evaluator, label, question) for evaluator in evaluators for label in labels for question in questions}
    if keys != expected:
        raise RuntimeError(f"Incomplete matrix: missing={sorted(expected - keys)[:8]}, extra={sorted(keys - expected)[:8]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--equal-harness-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    questions = [json.loads(line)["id"] for line in (args.equal_harness_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    bare_answers = normalize(load_jsonl(args.out_dir / "model_answers_bare.jsonl"), BARE_LABELS)
    prompt_answers = normalize(load_jsonl(args.equal_harness_dir / "model_answers.jsonl"), PROMPT_LABELS)
    answers = bare_answers + prompt_answers
    require_matrix(answers, BARE_LABELS | PROMPT_LABELS, questions)
    write_jsonl(args.out_dir / "model_answers.jsonl", answers)

    evaluators = None
    for round_number in (1, 2, 3):
        bare = normalize(load_jsonl(args.out_dir / f"model_scores_bare_round{round_number}.jsonl"), BARE_LABELS)
        prompt = normalize(load_jsonl(args.equal_harness_dir / f"model_scores_round{round_number}.jsonl"), PROMPT_LABELS)
        prompt = [row for row in prompt if row.get("evaluator_label") != "Qwen3-4B-prompt-RAG"]
        rows = bare + prompt
        if evaluators is None:
            evaluators = sorted({row["evaluator_label"] for row in rows})
        require_matrix(rows, BARE_LABELS | PROMPT_LABELS, questions, evaluators)
        write_jsonl(args.out_dir / f"model_scores_round{round_number}.jsonl", rows)

    print(f"Prepared {len(answers)} answers and three complete reviewer matrices in {args.out_dir}")


if __name__ == "__main__":
    main()
