from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DIMENSIONS = {
    "SR": "scenario relevance",
    "CSC": "citation support and scientific caution",
    "DQ": "design quality",
    "CS": "clarity and structure",
    "QR": "question responsiveness",
    "IS": "insightfulness",
}


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else (default or "")


def prompt_score(dim: str, description: str) -> float:
    while True:
        raw = input(f"  {dim} ({description}, 1-5): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("    Please enter a number from 1 to 5.")
            continue
        if 1.0 <= value <= 5.0:
            return value
        print("    Score must be between 1 and 5.")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(scores_path: Path, summary_path: Path) -> list[dict]:
    rows = load_jsonl(scores_path)
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        candidate = row["candidate_label"]
        scores = row["scores"]
        for dim in DIMENSIONS:
            grouped[candidate][dim].append(float(scores[dim]))

    summary_rows = []
    for candidate, dim_scores in grouped.items():
        rec = {"candidate_label": candidate}
        total = 0.0
        for dim in DIMENSIONS:
            value = mean(dim_scores[dim])
            rec[dim] = round(value, 6)
            total += value
        rec["Average"] = round(total / len(DIMENSIONS), 6)
        rec["Total_30"] = round(total, 6)
        rec["n_score_records"] = len(next(iter(dim_scores.values()))) if dim_scores else 0
        summary_rows.append(rec)

    summary_rows.sort(key=lambda r: r["Average"], reverse=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["candidate_label", *DIMENSIONS.keys(), "Average", "Total_30", "n_score_records"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_rows


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("\nNo score records found.")
        return

    print("\nCurrent summary:")
    header = ["candidate_label", "Average", "Total_30", "n_score_records"]
    widths = {key: max(len(key), *(len(str(row[key])) for row in rows)) for key in header}
    print("  ".join(key.ljust(widths[key]) for key in header))
    print("  ".join("-" * widths[key] for key in header))
    for row in rows:
        print("  ".join(str(row[key]).ljust(widths[key]) for key in header))


def interactive_loop(scores_path: Path, summary_path: Path) -> None:
    print("RAMAD Comment 8 interactive scoring utility")
    print("Enter one score record at a time. Press Ctrl+C to exit.\n")

    last_evaluator = ""
    last_question = ""
    while True:
        evaluator = prompt_text("Evaluator label", last_evaluator or None)
        candidate = prompt_text("Candidate/model label")
        question_id = prompt_text("Question ID", last_question or None)

        print("Scores:")
        scores = {dim: prompt_score(dim, desc) for dim, desc in DIMENSIONS.items()}
        rationale = prompt_text("Optional short rationale", "")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "evaluator_label": evaluator,
            "candidate_label": candidate,
            "question_id": question_id,
            "scores": scores,
            "rationale": rationale,
        }
        append_jsonl(scores_path, row)
        summary_rows = summarize(scores_path, summary_path)

        print(f"\nSaved score record to: {scores_path}")
        print(f"Updated summary table: {summary_path}")
        print_summary(summary_rows)

        last_evaluator = evaluator
        last_question = question_id
        again = prompt_text("\nAdd another record? y/n", "y").lower()
        if again not in {"y", "yes"}:
            break
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAMAD Comment 8 scoring utility.")
    parser.add_argument("--scores", default="interactive_scores.jsonl", help="Output JSONL score records.")
    parser.add_argument("--summary", default="interactive_score_summary.csv", help="Output summary CSV.")
    parser.add_argument("--summarize-only", action="store_true", help="Only summarize an existing JSONL score file.")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    summary_path = Path(args.summary)

    if args.summarize_only:
        rows = summarize(scores_path, summary_path)
        print_summary(rows)
        print(f"\nWrote summary table: {summary_path}")
        return

    try:
        interactive_loop(scores_path, summary_path)
    except KeyboardInterrupt:
        print("\nInterrupted. Existing records remain saved.")


if __name__ == "__main__":
    main()

