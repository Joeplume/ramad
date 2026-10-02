"""Build, score, and export the three-candidate RAMAD equal-harness control."""
import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_benchmark as rb
from reproduce_historical_scores import DIMENSIONS, write_csv

PANELS = {
    "equal": ["RAMAD", "Kimi-K3-prompt-RAG", "GPT-5.5-prompt-RAG"],
    "transfer": ["Kimi-K3-bare", "Kimi-K3-prompt-RAG", "GPT-5.5-bare", "GPT-5.5-prompt-RAG"],
}


def canonical(label, candidates):
    clean = re.sub(r"[^a-z0-9]", "", label.lower())
    lookup = {re.sub(r"[^a-z0-9]", "", name.lower()): name for name in candidates}
    return lookup.get(clean)


def parse_table(text, candidates):
    result = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip().replace("**", "").replace("`", "")
                 for c in line.strip().strip("|").split("|")]
        if len(cells) != 8:
            continue
        label = canonical(cells[0], candidates)
        if label is None or not all(re.fullmatch(r"\d+(?:\.\d+)?", c) for c in cells[1:]):
            continue
        values = [float(c) for c in cells[1:]]
        if not all(1 <= value <= 5 for value in values[:6]):
            raise ValueError(f"Out-of-range dimension score: {label}")
        result[label] = {
            "candidate": label,
            **dict(zip(DIMENSIONS, values[:6])),
            "reported_total": values[6],
            "dimension_sum": sum(values[:6]),
        }
    if set(result) != set(candidates):
        raise ValueError(f"Incomplete table: {sorted(set(candidates) - set(result))}")
    return [result[name] for name in candidates]


def build_request(answers_file, candidates):
    question = rb.load_jsonl(HERE / "historical/question_original.jsonl")[0]["question"]
    historical = {row["candidate_label"]: row for row in
                  rb.load_jsonl(HERE / "historical/archived_answers.jsonl")}
    generated = {row["candidate_label"]: row for row in rb.load_jsonl(answers_file)}
    required = set(candidates) - {"RAMAD"}
    if not required.issubset(generated):
        raise ValueError(f"Missing answers: {sorted(required - set(generated))}")

    source = (HERE / "historical/scoring_request_original_cn.txt").read_text(encoding="utf-8")
    prefix = source.split("评估模型列表：", 1)[0]
    count_cn = "三个" if len(candidates) == 3 else "四个"
    prefix = prefix.replace("以下 6 个", f"以下 {len(candidates)} 个").replace("以下六个模型", f"以下{count_cn}模型")
    lines = [prefix.rstrip(), "", "评估模型列表：", *[f"- {name}" for name in candidates],
             "", "输出表格格式如下：", "",
             "| 模型名称 | SR | CSC | DQ | CS | QR | IS | 总分 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|",
             *[f"| {name} | | | | | | | |" for name in candidates],
             "", "请严格只输出一个完整 Markdown 评分表；各维度为 1-5 分，总分为六维之和。",
             "", "Question：" + question]
    if "RAMAD" in candidates:
        lines += ["", "RAMAD：", historical["RAMAD"]["answer_and_appended_sources"].strip()]
    for name in candidates:
        if name == "RAMAD":
            continue
        row = generated[name]
        if row["question"] != question or row.get("finish_reason") not in ("stop", "eos_token"):
            raise ValueError(f"Invalid answer: {name}")
        passages = row.get("retrieved_passages", [])
        if row.get("condition") == "prompt_rag" and len(passages) != 5:
            raise ValueError(f"Expected five passages: {name}")
        lines += ["", f"{name}：", row["answer"].strip()]
        if passages:
            lines += ["", "参考文献（检索系统附加内容）："]
        for index, passage in enumerate(passages, 1):
            lines += [f"[{index}] {passage.get('title') or passage['source_id']}", passage["text"].strip()]
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "score", "export"])
    parser.add_argument("--answers-file", type=Path, required=True)
    parser.add_argument("--review-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel", choices=sorted(PANELS), default="equal")
    args = parser.parse_args()
    candidates = PANELS[args.panel]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    request = build_request(args.answers_file, candidates)
    request_path = args.output_dir / "joint_request.txt"
    if request_path.exists() and request_path.read_text(encoding="utf-8") != request:
        raise ValueError("Existing joint request differs from the current request")
    request_path.write_text(request, encoding="utf-8")

    config = rb.load_json(args.review_config)
    reviewers = rb.enabled_models(config, "evaluator_models", None)
    reviewers = [reviewer for reviewer in reviewers if reviewer["label"] != "Qwen3-4b"]
    messages = [{"role": "user", "content": request}]
    request_hash = rb.digest(messages)
    log_path = args.output_dir / "raw_reviews.jsonl"

    if args.command == "build":
        print(json.dumps({"candidates": candidates, "reviewers": [r["label"] for r in reviewers],
                          "planned_calls": len(reviewers) * 3, "request_sha256": request_hash},
                         ensure_ascii=False, indent=2))
        return

    existing = rb.load_jsonl(log_path) if log_path.exists() else []
    done = {(row["reviewer"], row["round"]) for row in existing}
    if args.command == "score":
        for reviewer in reviewers:
            for round_id in range(1, 4):
                if (reviewer["label"], round_id) in done:
                    continue
                response, profile, parameters = rb.completion(config, reviewer, messages, "scoring")
                parsed = parse_table(rb.visible_text(response), candidates)
                rb.append_jsonl(log_path, {
                    "run_label": f"ramad-{args.panel}-panel-20261002",
                    "timestamp_utc": rb.utc_now(),
                    "reviewer": reviewer["label"],
                    "round": round_id,
                    "request_sha256": request_hash,
                    "requested_model": reviewer["model"],
                    "returned_model": response.get("model", ""),
                    "api_profile": profile,
                    "parameters": rb.saved_parameters(parameters),
                    "response": response,
                    "status": "complete",
                    "parsed_scores": parsed,
                })
                print(f"saved {reviewer['label']} round {round_id}", flush=True)

    records = rb.load_jsonl(log_path)
    expected = {(reviewer["label"], round_id) for reviewer in reviewers for round_id in (1, 2, 3)}
    actual = {(row["reviewer"], row["round"]) for row in records}
    if actual != expected:
        raise ValueError(f"Reviewer matrix incomplete: {sorted(expected - actual)}")
    rows = [{"reviewer": row["reviewer"], "round": row["round"], **score}
            for row in records for score in row["parsed_scores"]]
    write_csv(args.output_dir / "reviewer_rounds.csv", rows)
    summary = []
    for candidate in candidates:
        subset = [row for row in rows if row["candidate"] == candidate]
        summary.append({
            "candidate": candidate,
            "n_ratings": len(subset),
            "mean_total": mean(float(row["dimension_sum"]) for row in subset),
            **{dimension: mean(float(row[dimension]) for row in subset) for dimension in DIMENSIONS},
        })
    write_csv(args.output_dir / "mean_scores.csv", summary)
    manifest = {
        "candidates": candidates,
        "reviewers": [reviewer["label"] for reviewer in reviewers],
        "rounds_per_reviewer": 3,
        "ratings_per_candidate": len(reviewers) * 3,
        "request_sha256": request_hash,
        "score_definition": "mean of the six dimension sums",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
