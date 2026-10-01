import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


DIMENSIONS = ["SR", "CSC", "DQ", "CS", "QR", "IS"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def enabled_labels(config: dict, key: str):
    return [m["label"] for m in config[key] if m.get("enabled", True)]


def minmax(v: pd.Series):
    v = v.astype(float)
    vmin, vmax = v.min(), v.max()
    if vmax == vmin:
        return pd.Series(np.ones_like(v.values) / len(v), index=v.index)
    return (v - vmin) / (vmax - vmin)


def renorm(z: pd.Series):
    s = float(z.sum())
    if s == 0:
        return pd.Series(np.ones_like(z.values) / len(z), index=z.index)
    return z / s


def iterative_weights(S: pd.DataFrame, candidate_labels: list, evaluator_labels: list, max_iters=200, tol=1e-7, epsilon=1e-6, damp=0.05):
    reviewers = [r for r in evaluator_labels if r in S.index]
    anchors = [r for r in reviewers if r in candidate_labels]
    if len(reviewers) == 0:
        raise ValueError("No evaluator rows found in score matrix")
    if len(anchors) < 2:
        uniform = pd.Series(1 / len(reviewers), index=reviewers)
        return uniform, pd.DataFrame([{"k": 0, "delta_L1": 0.0, **{f"alpha_{r}": float(uniform[r]) for r in reviewers}}])

    alpha = pd.Series(1 / len(reviewers), index=reviewers)
    history = []
    S_work = S.loc[reviewers, candidate_labels].astype(float)
    for k in range(1, max_iters + 1):
        model_scores = (alpha @ S_work).rename("Score")
        anchor_scores = model_scores[anchors]
        z = minmax(anchor_scores) + epsilon
        z = renorm(z)
        z = z.reindex(reviewers).fillna(0.0)
        uniform = pd.Series(1 / len(reviewers), index=reviewers)
        alpha_new = renorm((1 - damp) * z + damp * uniform)
        delta = float(np.abs(alpha_new - alpha).sum())
        row = {"k": k, "delta_L1": delta}
        row.update({f"alpha_{r}": float(alpha_new[r]) for r in reviewers})
        row.update({f"Score_{c}": float(model_scores[c]) for c in candidate_labels})
        history.append(row)
        alpha = alpha_new
        if delta < tol:
            break
    return alpha, pd.DataFrame(history)


def aggregate(config_path: Path, scores_path: Path, out_dir: Path):
    config = load_json(config_path)
    candidate_labels = enabled_labels(config, "candidate_models")
    evaluator_labels = enabled_labels(config, "evaluator_models")
    rows = load_jsonl(scores_path)

    flat = []
    errors = []
    for row in rows:
        if row.get("parse_error"):
            errors.append(row)
            continue
        scores = row.get("scores") or {}
        if not all(dim in scores for dim in DIMENSIONS):
            errors.append(row)
            continue
        for dim in DIMENSIONS:
            flat.append({
                "evaluator": row["evaluator_label"],
                "candidate": row["candidate_label"],
                "question_id": row["question_id"],
                "dimension": dim,
                "score": float(scores[dim]),
            })

    if not flat:
        raise RuntimeError("No valid score rows found")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(flat)
    df.to_csv(out_dir / "all_scores_long.csv", index=False, encoding="utf-8-sig")

    simple = (
        df.groupby(["candidate", "dimension"], as_index=False)["score"]
        .mean()
        .pivot(index="candidate", columns="dimension", values="score")
        .reindex(candidate_labels)
    )
    simple["Average"] = simple[DIMENSIONS].mean(axis=1)
    simple["Total_30"] = simple[DIMENSIONS].sum(axis=1)
    simple.sort_values("Average", ascending=False).to_csv(out_dir / "simple_mean_scores.csv", encoding="utf-8-sig")

    # Mean score by evaluator, candidate, and dimension across all question variants.
    edc = (
        df.groupby(["evaluator", "candidate", "dimension"], as_index=False)["score"]
        .mean()
    )

    dim_scores = {}
    dim_weights = {}
    histories = {}
    for dim in DIMENSIONS:
        sub = edc[edc["dimension"] == dim].pivot(index="evaluator", columns="candidate", values="score")
        sub = sub.reindex(index=evaluator_labels, columns=candidate_labels)
        alpha, hist = iterative_weights(sub, candidate_labels, evaluator_labels)
        final = (alpha @ sub.loc[alpha.index, candidate_labels]).rename(dim)
        dim_scores[dim] = final
        dim_weights[dim] = alpha
        histories[dim] = hist
        hist.to_csv(out_dir / f"history_{dim}.csv", index=False, encoding="utf-8-sig")

    weighted = pd.DataFrame(dim_scores)
    weighted["Average"] = weighted[DIMENSIONS].mean(axis=1)
    weighted["Total_30"] = weighted[DIMENSIONS].sum(axis=1)
    weighted = weighted.sort_values("Average", ascending=False)
    weighted.to_csv(out_dir / "weighted_final_scores.csv", encoding="utf-8-sig")

    weights = pd.DataFrame(dim_weights).T
    weights.to_csv(out_dir / "evaluator_weights_by_dimension.csv", encoding="utf-8-sig")

    if errors:
        with (out_dir / "score_parse_errors.jsonl").open("w", encoding="utf-8") as f:
            for row in errors:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Compact manuscript-ready table.
    summary = weighted.reset_index().rename(columns={"index": "Model"})
    summary.to_csv(out_dir / "table_s4_updated.csv", index=False, encoding="utf-8-sig")

    print("Updated weighted ranking:")
    print(weighted.round(3).to_string())
    print(f"\nWrote outputs to: {out_dir}")
    if errors:
        print(f"Warning: {len(errors)} score rows had parse errors or missing dimensions.")


def main():
    parser = argparse.ArgumentParser(description="Aggregate frontier-model comparison scores")
    parser.add_argument("--config", default="run_config.json")
    parser.add_argument("--scores", default="outputs/model_scores.jsonl")
    parser.add_argument("--out-dir", default="outputs/summary")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    aggregate((root / args.config).resolve(), (root / args.scores).resolve(), (root / args.out_dir).resolve())


if __name__ == "__main__":
    main()
