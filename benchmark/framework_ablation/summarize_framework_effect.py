"""Calculate paired prompt-RAG minus bare effects for Claude and Kimi."""

import argparse
import csv
import itertools
import json
from pathlib import Path


PAIRS = (
    ("Claude Fable 5", "Claude-Fable-5-bare", "Claude-Fable-5-prompt-RAG"),
    ("Kimi K3", "Kimi-K3-bare", "Kimi-K3-prompt-RAG"),
)


def percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()

    with (args.summary_dir / "weighted_final_scores.csv").open(encoding="utf-8-sig", newline="") as handle:
        totals = {row["Model"]: float(row["Total_30"]) for row in csv.DictReader(handle)}
    with (args.summary_dir / "per_question_weighted_scores.csv").open(encoding="utf-8-sig", newline="") as handle:
        per_question = {(row["Model"], row["question_id"]): float(row["Total_30"]) for row in csv.DictReader(handle)}

    rows = []
    for model, bare_label, enhanced_label in PAIRS:
        questions = sorted(question for candidate, question in per_question if candidate == bare_label)
        differences = [per_question[(enhanced_label, question)] - per_question[(bare_label, question)] for question in questions]
        bootstrap = [sum(differences[index] for index in sample) / len(sample) for sample in itertools.product(range(len(differences)), repeat=len(differences))]
        observed = abs(sum(differences) / len(differences))
        sign_flip = [abs(sum(sign * value for sign, value in zip(signs, differences)) / len(differences)) for signs in itertools.product((-1, 1), repeat=len(differences))]
        bare = totals[bare_label]
        enhanced = totals[enhanced_label]
        difference = enhanced - bare
        rows.append({
            "model": model,
            "bare_score_30": bare,
            "prompt_rag_score_30": enhanced,
            "absolute_gain_30": difference,
            "relative_gain_percent": 100 * difference / bare,
            "paired_question_mean_gain_30": sum(differences) / len(differences),
            "bootstrap_95_low": percentile(bootstrap, 0.025),
            "bootstrap_95_high": percentile(bootstrap, 0.975),
            "exact_sign_flip_p": sum(value >= observed - 1e-12 for value in sign_flip) / len(sign_flip),
            "n_questions": len(questions),
        })

    output = args.summary_dir / "framework_effects.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.summary_dir / "framework_effects.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
