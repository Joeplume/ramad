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

GENERATION_SYSTEM_PROMPT = """You are an expert in Raman/SERS analytical chemistry, aquaculture-drug residue detection, chemometrics, and field-deployable sensing systems.

Answer the question as a technically rigorous research assistant. Focus on realistic experimental choices, sample preparation, Raman/SERS acquisition, spectral analysis, limitations, and field constraints. Avoid unsupported performance claims and invented citations. Use concise, structured prose that a researcher can evaluate.
"""

SCORING_SYSTEM_PROMPT = """You are an impartial reviewer of LLM-generated Raman/SERS workflow recommendations.

Score the candidate answer from 1 to 5 on each dimension, where 1 = poor, 3 = acceptable, and 5 = excellent.

- SR: scenario relevance
- CSC: scientific caution and support
- DQ: design quality
- CS: clarity and structure
- QR: question responsiveness
- IS: insightfulness

Return only strict JSON with this structure:
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    for attempt in range(8):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.5 * (attempt + 1))


def enabled_models(config: dict, key: str, labels=None):
    models = [item for item in config.get(key, []) if item.get("enabled", False)]
    if labels:
        wanted = set(labels)
        models = [item for item in models if item.get("label") in wanted]
        missing = wanted - {item.get("label") for item in models}
        if missing:
            raise ValueError(f"Requested labels are not enabled in {key}: {sorted(missing)}")
    return models


def api_profile(config: dict, model_config: dict):
    profile_name = model_config.get("api_profile", "default")
    profiles = config.get("api_profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Unknown api_profile '{profile_name}' for {model_config.get('label', 'model')}")
    profile = dict(profiles[profile_name])
    profile["name"] = profile_name
    return profile


def api_request(profile: dict, method: str, route: str, payload=None):
    base_url = str(profile.get("base_url", "")).rstrip("/")
    if not base_url:
        raise ValueError(f"Missing base_url for API profile '{profile['name']}'")

    api_key_env = str(profile.get("api_key_env", "")).strip()
    api_key = os.environ.get(api_key_env) if api_key_env else ""
    if profile.get("require_api_key", True) and not api_key:
        raise RuntimeError(f"Missing API key environment variable: {api_key_env}")

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    timeout_s = int(profile.get("request_timeout_s", 180))
    max_retries = int(profile.get("max_retries", 3))
    retry_sleep_s = float(profile.get("retry_sleep_s", 4.0))

    for attempt in range(max_retries + 1):
        request = urllib.request.Request(base_url + route, data=body, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            exc.read()
            if not retryable or attempt >= max_retries:
                raise RuntimeError(f"API request failed with HTTP status {exc.code}") from exc
        except (urllib.error.URLError, http.client.RemoteDisconnected, ssl.SSLError, TimeoutError, ConnectionError) as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"API request failed: {type(exc).__name__}") from exc
        wait_s = retry_sleep_s * (attempt + 1)
        print(f"Transient API error; retrying in {wait_s:.1f} s", file=sys.stderr)
        time.sleep(wait_s)
    raise RuntimeError("API request failed")


def visible_text(response: dict):
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        item_type = content.get("type")
        if item_type in {None, "text", "output_text"} and isinstance(content.get("text"), str):
            return content["text"]
        return ""
    if not isinstance(content, list):
        return ""
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {None, "text", "output_text"} and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "".join(parts)


def finish_reason(response: dict):
    choices = response.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("finish_reason") or "")


def usage_counts(response: dict):
    usage = response.get("usage") or {}
    allowed = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {key: usage[key] for key in allowed if isinstance(usage.get(key), (int, float))}


def saved_parameters(parameters: dict):
    allowed = (
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "frequency_penalty",
        "presence_penalty",
    )
    return {key: parameters[key] for key in allowed if key in parameters}


def completion(config: dict, model_config: dict, messages: list, parameter_group: str):
    profile = api_profile(config, model_config)
    parameters = dict(config.get(parameter_group, {}))
    parameters.update(model_config.get("request_overrides", {}))
    reserved = {"model", "messages"} & set(parameters)
    if reserved:
        raise ValueError(f"Reserved request parameter(s): {sorted(reserved)}")
    if parameters.get("stream"):
        raise ValueError("Streaming responses are not supported by this runner")
    payload = {"model": model_config["model"], "messages": messages, **parameters}
    response = api_request(profile, "POST", profile.get("chat_completions_path", "/v1/chat/completions"), payload)
    return response, profile["name"], parameters


def existing_keys(path: Path, fields: list):
    if not path.exists():
        return set()
    return {tuple(row.get(field) for field in fields) for row in load_jsonl(path)}


def parse_score(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    scores = {}
    for dimension in DIMENSIONS:
        value = float(data[dimension])
        if not 1.0 <= value <= 5.0:
            raise ValueError(f"{dimension} is outside the 1-5 range")
        scores[dimension] = value
    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return scores, rationale


def list_models(config: dict, profile_name: str):
    profiles = config.get("api_profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Unknown API profile: {profile_name}")
    profile = dict(profiles[profile_name])
    profile["name"] = profile_name
    response = api_request(profile, "GET", profile.get("models_path", "/v1/models"))
    compact = []
    for item in response.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            compact.append({"id": item["id"]})
    print(json.dumps({"data": compact}, ensure_ascii=False, indent=2))


def dry_run(config: dict, questions: list, candidate_labels, evaluator_labels):
    candidates = enabled_models(config, "candidate_models", candidate_labels)
    evaluators = enabled_models(config, "evaluator_models", evaluator_labels)
    plan = {
        "run_label": config.get("run_label", ""),
        "question_count": len(questions),
        "question_ids": [row.get("id") for row in questions],
        "enabled_candidates": [
            {"label": item["label"], "model": item["model"], "api_profile": item.get("api_profile", "default")}
            for item in candidates
        ],
        "enabled_evaluators": [
            {"label": item["label"], "model": item["model"], "api_profile": item.get("api_profile", "default")}
            for item in evaluators
        ],
        "planned_generation_calls": len(questions) * len(candidates),
        "planned_scoring_calls_after_full_generation": len(questions) * len(candidates) * len(evaluators),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def generate(config: dict, questions: list, output_path: Path, sleep_s: float, labels):
    candidates = enabled_models(config, "candidate_models", labels)
    if not candidates:
        raise RuntimeError("No candidate model is enabled")
    done = existing_keys(output_path, ["run_label", "candidate_label", "question_id"])
    run_label = config.get("run_label", "")
    for model_config in candidates:
        for question in questions:
            key = (run_label, model_config["label"], question["id"])
            if key in done:
                print(f"skip existing: {model_config['label']} / {question['id']}")
                continue
            messages = [
                {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": question["question"]},
            ]
            print(f"generate: {model_config['label']} / {question['id']}")
            started = time.monotonic()
            response, profile_name, parameters = completion(config, model_config, messages, "generation")
            answer = visible_text(response)
            row = {
                "schema_version": 1,
                "record_type": "candidate_answer",
                "timestamp_utc": utc_now(),
                "run_label": run_label,
                "candidate_label": model_config["label"],
                "candidate_model": model_config["model"],
                "api_profile": profile_name,
                "question_id": question["id"],
                "topic": question.get("topic", ""),
                "question": question["question"],
                "answer": answer,
                "request_parameters": saved_parameters(parameters),
                "finish_reason": finish_reason(response),
                "usage": usage_counts(response),
                "latency_s": round(time.monotonic() - started, 3),
            }
            append_jsonl(output_path, row)
            time.sleep(sleep_s)


def score(config: dict, answer_path: Path, score_path: Path, sleep_s: float, labels):
    evaluators = enabled_models(config, "evaluator_models", labels)
    if not evaluators:
        raise RuntimeError("No evaluator model is enabled")
    answers = load_jsonl(answer_path)
    done = existing_keys(score_path, ["run_label", "evaluator_label", "candidate_label", "question_id"])
    for evaluator in evaluators:
        for answer_row in answers:
            run_label = answer_row.get("run_label", config.get("run_label", ""))
            key = (run_label, evaluator["label"], answer_row["candidate_label"], answer_row["question_id"])
            if key in done:
                print(f"skip existing: {evaluator['label']} -> {answer_row['candidate_label']} / {answer_row['question_id']}")
                continue
            user_prompt = (
                f"Question:\n{answer_row['question']}\n\n"
                f"Candidate model label:\n{answer_row['candidate_label']}\n\n"
                f"Candidate answer:\n{answer_row['answer']}"
            )
            messages = [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            print(f"score: {evaluator['label']} -> {answer_row['candidate_label']} / {answer_row['question_id']}")
            started = time.monotonic()
            response, profile_name, parameters = completion(config, evaluator, messages, "scoring")
            text = visible_text(response)
            try:
                scores, rationale = parse_score(text)
                parse_error = ""
            except Exception as exc:
                scores, rationale = {}, ""
                parse_error = f"{type(exc).__name__}: evaluator output was not valid six-dimension JSON"
            row = {
                "schema_version": 1,
                "record_type": "evaluator_score",
                "timestamp_utc": utc_now(),
                "run_label": run_label,
                "evaluator_label": evaluator["label"],
                "evaluator_model": evaluator["model"],
                "api_profile": profile_name,
                "candidate_label": answer_row["candidate_label"],
                "candidate_model": answer_row.get("candidate_model", ""),
                "question_id": answer_row["question_id"],
                "scores": scores,
                "rationale": rationale,
                "parse_error": parse_error,
                "request_parameters": saved_parameters(parameters),
                "finish_reason": finish_reason(response),
                "usage": usage_counts(response),
                "latency_s": round(time.monotonic() - started, 3),
            }
            append_jsonl(score_path, row)
            time.sleep(sleep_s)


def resolve(root: Path, value: str):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate and score Comment 8 benchmark answers through OpenAI-compatible APIs."
    )
    parser.add_argument("command", choices=["dry-run", "list-models", "generate", "score"])
    parser.add_argument("--config", default="benchmark_config.json")
    parser.add_argument("--questions", default="questions.jsonl")
    parser.add_argument("--answers", default="outputs/model_answers.jsonl")
    parser.add_argument("--scores", default="outputs/model_scores.jsonl")
    parser.add_argument("--profile", default="default", help="API profile used by list-models")
    parser.add_argument("--candidate-label", action="append", default=[])
    parser.add_argument("--evaluator-label", action="append", default=[])
    parser.add_argument("--sleep-s", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_json(resolve(root, args.config))
    questions_path = resolve(root, args.questions)
    questions = load_jsonl(questions_path) if args.command != "list-models" else []

    if args.command == "dry-run":
        dry_run(config, questions, args.candidate_label, args.evaluator_label)
    elif args.command == "list-models":
        list_models(config, args.profile)
    elif args.command == "generate":
        generate(config, questions, resolve(root, args.answers), args.sleep_s, args.candidate_label)
    elif args.command == "score":
        score(config, resolve(root, args.answers), resolve(root, args.scores), args.sleep_s, args.evaluator_label)


if __name__ == "__main__":
    main()
