"""Average repeated blind reviewer calls before applying the paper's weighting method."""

import argparse
import json
from pathlib import Path


DIMENSIONS = ("SR", "CSC", "DQ", "CS", "QR", "IS")
KEYS = ("run_label", "evaluator_label", "candidate_label", "question_id")


def load_round(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = tuple(row.get(field) for field in KEYS)
        if key in rows:
            raise ValueError(f"Duplicate reviewer-question pair in {path}: {key}")
        if row.get("finish_reason") != "stop" or row.get("parse_error"):
            raise ValueError(f"Incomplete reviewer response in {path}: {key}")
        if any(not isinstance(row.get("scores", {}).get(dim), (int, float)) for dim in DIMENSIONS):
            raise ValueError(f"Missing score dimension in {path}: {key}")
        if any(not 1 <= float(row["scores"][dim]) <= 5 for dim in DIMENSIONS):
            raise ValueError(f"Score outside 1-5 range in {path}: {key}")
        rows[key] = row
    return rows


def combine(paths: list[Path], output: Path) -> None:
    if len(paths) != 3:
        raise ValueError("Exactly three reviewer rounds are required by the manuscript protocol")
    rounds = [load_round(path) for path in paths]
    keys = set(rounds[0])
    for path, rows in zip(paths[1:], rounds[1:]):
        if set(rows) != keys:
            raise ValueError(f"Reviewer matrix differs in {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for key in sorted(keys):
            items = [rows[key] for rows in rounds]
            if len({row.get("blind_response_id") for row in items}) != 1:
                raise ValueError(f"Blinded answer differs across rounds: {key}")
            if any(row.get("messages") != items[0].get("messages") for row in items[1:]):
                raise ValueError(f"Reviewer prompt differs across rounds: {key}")
            result = dict(items[0])
            result["record_type"] = "mean_evaluator_score"
            result["scores"] = {
                dim: sum(float(row["scores"][dim]) for row in items) / len(items)
                for dim in DIMENSIONS
            }
            result["rationale"] = "Mean of independent blinded reviewer calls; raw rationales remain in the round files."
            result["review_round_count"] = len(items)
            result["round_sources"] = [path.name for path in paths]
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Combined {len(paths)} rounds and {len(keys)} reviewer-question pairs: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    combine(args.rounds, args.out)
