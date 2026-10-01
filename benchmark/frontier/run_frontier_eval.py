import argparse
import http.client
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DIMENSIONS = ["SR", "CSC", "DQ", "CS", "QR", "IS"]

GEN_SYSTEM_PROMPT = """You are an expert in Raman/SERS analytical chemistry, aquaculture-drug residue detection, chemometrics, and field-deployable sensing systems.

Answer the user question as a technically rigorous research assistant. Requirements:
- Focus on aquaculture drugs such as malachite green (MG), sulfathiazole (STZ), and ofloxacin (OFX) where relevant.
- Include practical experimental details, realistic SERS substrate/additive choices, preprocessing, Raman acquisition, data analysis, validation, and limitations.
- Avoid unsupported claims, unrealistic detection promises, and vague generic advice.
- Prefer concise sectioned prose with enough implementation detail for a researcher to evaluate the workflow.
"""

SCORING_SYSTEM_PROMPT = """You are an impartial reviewer for an Analytical Chemistry manuscript evaluating LLM-generated SERS workflow recommendations.

Score the candidate answer on six dimensions from 1 to 5, where 1 = poor, 3 = acceptable, and 5 = excellent.

Dimensions:
- SR: Scenario relevance. Does the answer address fishpond/aquaculture-drug Raman/SERS detection under realistic field or low-SNR conditions?
- CSC: Citation support and scientific caution. Does the answer avoid hallucinated certainty, acknowledge limits, and ground claims in plausible analytical chemistry knowledge?
- DQ: Design quality. Are the proposed experimental/SERS/modeling choices coherent, feasible, and technically detailed?
- CS: Clarity and structure. Is the answer organized, precise, and easy to evaluate?
- QR: Question responsiveness. Does the answer directly answer all parts of the prompt?
- IS: Insightfulness. Does the answer provide nontrivial methodological insight or innovation beyond generic statements?

Return only strict JSON with this schema:
{
  "SR": <number>,
  "CSC": <number>,
  "DQ": <number>,
  "CS": <number>,
  "QR": <number>,
  "IS": <number>,
  "rationale": "one concise paragraph"
}
"""


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


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(row, ensure_ascii=False) + "\n"
    for attempt in range(8):
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(text)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(1.5 * (attempt + 1))


def enabled_models(config: dict, key: str):
    return [m for m in config[key] if m.get("enabled", True)]


def api_request(config: dict, method: str, path: str, payload=None):
    api_key = os.environ.get(config.get("api_key_env", "QINGYUN_API_KEY"))
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {config.get('api_key_env', 'QINGYUN_API_KEY')}")

    base_url = config["base_url"].rstrip("/")
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    timeout = int(config.get("request_timeout_s", 180))
    max_retries = int(config.get("max_retries", 4))
    retry_sleep_s = float(config.get("retry_sleep_s", 8))
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {e.code} from {path}: {body}")
            retryable = e.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                raise last_error from e
        except urllib.error.URLError as e:
            last_error = RuntimeError(f"URL error from {path}: {e}")
            if attempt >= max_retries:
                raise last_error from e
        except (http.client.RemoteDisconnected, ssl.SSLError, TimeoutError, ConnectionError) as e:
            last_error = RuntimeError(f"Connection error from {path}: {type(e).__name__}: {e}")
            if attempt >= max_retries:
                raise last_error from e
        wait_s = retry_sleep_s * (attempt + 1)
        print(f"retrying after transient API error in {wait_s:.1f}s: {last_error}", file=sys.stderr)
        time.sleep(wait_s)


def list_models(config_path: Path):
    config = load_json(config_path)
    data = api_request(config, "GET", "/v1/models")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def chat_completion(config: dict, model_cfg: dict, messages: list, mode: str):
    params = dict(config.get(mode, {}))
    payload = {
        "model": model_cfg["model"],
        "messages": messages,
        **params,
    }
    if model_cfg.get("extra_body"):
        payload.update(model_cfg["extra_body"])
    result = api_request(config, "POST", "/v1/chat/completions", payload)
    return result


def extract_text(result: dict):
    choices = result.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return content or ""


def existing_keys(path: Path, fields: list):
    keys = set()
    if not path.exists():
        return keys
    for row in load_jsonl(path):
        keys.add(tuple(row.get(f) for f in fields))
    return keys


def generate_outputs(config_path: Path, questions_path: Path, out_path: Path, sleep_s: float):
    config = load_json(config_path)
    questions = load_jsonl(questions_path)
    done = existing_keys(out_path, ["candidate_label", "question_id"])
    for model_cfg in enabled_models(config, "candidate_models"):
        for q in questions:
            key = (model_cfg["label"], q["id"])
            if key in done:
                print(f"skip generated {key}")
                continue
            messages = [
                {"role": "system", "content": GEN_SYSTEM_PROMPT},
                {"role": "user", "content": q["question"]},
            ]
            print(f"generating {model_cfg['label']} / {q['id']}")
            started = time.time()
            result = chat_completion(config, model_cfg, messages, "generation")
            row = {
                "run_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_label": model_cfg["label"],
                "candidate_model": model_cfg["model"],
                "question_id": q["id"],
                "topic": q.get("topic", ""),
                "question": q["question"],
                "answer": extract_text(result),
                "latency_s": round(time.time() - started, 3),
                "raw_response": result,
            }
            append_jsonl(out_path, row)
            time.sleep(sleep_s)


def parse_score_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
        for dim in DIMENSIONS:
            m = re.search(rf'"?{dim}"?\s*:\s*([1-5](?:\.\d+)?)', text)
            if m:
                data[dim] = float(m.group(1))
        if len(data) != len(DIMENSIONS):
            raise
        data["rationale"] = "Parsed numeric scores from non-strict JSON response."
    scores = {}
    for dim in DIMENSIONS:
        val = float(data[dim])
        if val < 1 or val > 5:
            raise ValueError(f"{dim} out of range: {val}")
        scores[dim] = val
    return scores, data.get("rationale", "")


def score_outputs(config_path: Path, outputs_path: Path, scores_path: Path, sleep_s: float):
    config = load_json(config_path)
    outputs = load_jsonl(outputs_path)
    done = existing_keys(scores_path, ["evaluator_label", "candidate_label", "question_id"])
    for evaluator in enabled_models(config, "evaluator_models"):
        for row in outputs:
            key = (evaluator["label"], row["candidate_label"], row["question_id"])
            if key in done:
                print(f"skip scored {key}")
                continue
            user_prompt = (
                "Question:\n" + row["question"] + "\n\n"
                "Candidate model label:\n" + row["candidate_label"] + "\n\n"
                "Candidate answer:\n" + row["answer"]
            )
            messages = [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            print(f"scoring {evaluator['label']} -> {row['candidate_label']} / {row['question_id']}")
            started = time.time()
            result = chat_completion(config, evaluator, messages, "scoring")
            text = extract_text(result)
            try:
                scores, rationale = parse_score_json(text)
                parse_error = ""
            except Exception as exc:
                scores, rationale = {}, ""
                parse_error = f"{type(exc).__name__}: {exc}; raw={text[:1000]}"
            out = {
                "run_utc": datetime.now(timezone.utc).isoformat(),
                "evaluator_label": evaluator["label"],
                "evaluator_model": evaluator["model"],
                "candidate_label": row["candidate_label"],
                "candidate_model": row["candidate_model"],
                "question_id": row["question_id"],
                "scores": scores,
                "rationale": rationale,
                "parse_error": parse_error,
                "latency_s": round(time.time() - started, 3),
                "raw_response": result,
            }
            append_jsonl(scores_path, out)
            time.sleep(sleep_s)


def main():
    parser = argparse.ArgumentParser(description="Frontier-model comparison experiment runner")
    parser.add_argument("command", choices=["list-models", "generate", "score"])
    parser.add_argument("--config", default="run_config.json")
    parser.add_argument("--questions", default="questions.jsonl")
    parser.add_argument("--outputs", default="outputs/model_outputs.jsonl")
    parser.add_argument("--scores", default="outputs/model_scores.jsonl")
    parser.add_argument("--sleep-s", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = (root / args.config).resolve()
    questions = (root / args.questions).resolve()
    outputs = (root / args.outputs).resolve()
    scores = (root / args.scores).resolve()

    if args.command == "list-models":
        list_models(config)
    elif args.command == "generate":
        generate_outputs(config, questions, outputs, args.sleep_s)
    elif args.command == "score":
        score_outputs(config, outputs, scores, args.sleep_s)


if __name__ == "__main__":
    main()
