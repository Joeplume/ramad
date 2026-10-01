import argparse
import hashlib
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
LOCAL_MODELS = {}

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
if not (PROMPT_DIR / "original_rag_qa_template_2025.txt").is_file():
    PROMPT_DIR = Path(__file__).resolve().parent
RAG_USER_TEMPLATE = "\n" + (PROMPT_DIR / "original_rag_qa_template_2025.txt").read_text(encoding="utf-8")
SCORING_SYSTEM_PROMPT = (PROMPT_DIR / "scoring_rubric_cn.txt").read_text(encoding="utf-8").strip()


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


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_contexts(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Frozen retrieval contexts are required: {path}")
    contexts = {}
    for row in load_jsonl(path):
        question_id = row["question_id"]
        if question_id in contexts:
            raise ValueError(f"Duplicate retrieval context for {question_id}")
        passages = row.get("passages") or []
        if not passages:
            raise ValueError(f"No retrieved passages for {question_id}")
        for passage in passages:
            if not all(str(passage.get(key, "")).strip() for key in ("source_id", "text")):
                raise ValueError(f"A passage for {question_id} lacks source_id or text")
        contexts[question_id] = passages
    return contexts


def build_messages(question, passages, condition):
    if condition == "bare":
        return [{"role": "user", "content": question}]
    if condition != "prompt_rag":
        raise ValueError(f"Unknown condition: {condition}")
    if not passages:
        raise ValueError("prompt_rag requires frozen retrieved passages")
    context = "\n\n".join(item["text"] for item in passages)
    return [{"role": "user", "content": RAG_USER_TEMPLATE.format(context=context, question=question)}]


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
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if not retryable or attempt >= max_retries:
                raise RuntimeError(f"API request failed with HTTP status {exc.code}: {detail}") from exc
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
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "output_config",
        "reasoning_effort",
        "thinking",
        "enable_thinking",
        "repetition_penalty",
        "do_sample",
        "prompt_format",
    )
    return {key: parameters[key] for key in allowed if key in parameters}


def completion(config: dict, model_config: dict, messages: list, parameter_group: str):
    if model_config.get("backend") == "local":
        return local_completion(config, model_config, messages, parameter_group)
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


def local_completion(config: dict, model_config: dict, messages: list, parameter_group: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model"]
    revision = model_config.get("revision", config.get("base_model_revision"))
    adapter_path = model_config.get("adapter_path")
    use_4bit = bool(model_config.get("use_4bit", False))
    tokenizer_use_fast = bool(model_config.get("tokenizer_use_fast", True))
    key = (model_name, revision, adapter_path, use_4bit, tokenizer_use_fast)
    if key not in LOCAL_MODELS:
        # Sequential benchmark candidates must fit on one GPU independently.
        # Keep at most one loaded model; cache reuse remains possible for the
        # RAG-only and bare conditions when their model/tokenizer keys match.
        LOCAL_MODELS.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        kwargs = {"device_map": "auto", "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        if use_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit local inference requires CUDA")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs.pop("torch_dtype")
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=tokenizer_use_fast)
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **kwargs)
        if adapter_path:
            if not Path(adapter_path).is_dir():
                raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        LOCAL_MODELS[key] = (tokenizer, model)
    tokenizer, model = LOCAL_MODELS[key]
    prompt_format = model_config.get("prompt_format", "chat")
    if prompt_format == "raw":
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Raw source inference requires exactly one user prompt")
        prompt = messages[0]["content"]
    elif prompt_format != "chat":
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    elif getattr(tokenizer, "chat_template", None):
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "enable_thinking" in model_config:
            template_kwargs["enable_thinking"] = bool(model_config["enable_thinking"])
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    else:
        prompt = "\n\n".join(f"{item['role']}:\n{item['content']}" for item in messages) + "\n\nassistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    parameters = dict(config.get(parameter_group, {}))
    parameters.update(model_config.get("request_overrides", {}))
    parameters["prompt_format"] = prompt_format
    if "enable_thinking" in model_config:
        parameters["enable_thinking"] = bool(model_config["enable_thinking"])
    temperature = float(parameters.get("temperature", 0))
    generation = {
        "max_new_tokens": int(parameters.get("max_tokens", 1024)),
        "do_sample": bool(parameters.get("do_sample", temperature > 0)),
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": float(parameters.get("repetition_penalty", 1.0)),
    }
    if temperature > 0:
        generation["temperature"] = temperature
    if "top_p" in parameters:
        generation["top_p"] = float(parameters["top_p"])
    if "top_k" in parameters:
        generation["top_k"] = int(parameters["top_k"])
    if "seed" in parameters:
        from transformers import set_seed
        set_seed(int(parameters["seed"]))
    effective_generation = model.generation_config.to_dict()
    effective_generation.update(generation)
    with torch.inference_mode():
        tokens = model.generate(**inputs, **generation)
    answer_ids = tokens[0, inputs["input_ids"].shape[1] :]
    raw_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    answer = raw_answer.split("</think>", 1)[1].strip() if "</think>" in raw_answer else raw_answer
    generation_limit = int(parameters.get("max_tokens", 1024))
    eos_ids = effective_generation.get("eos_token_id") or []
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    ended_with_eos = len(answer_ids) > 0 and int(answer_ids[-1]) in eos_ids
    reached_limit = answer_ids.shape[0] >= generation_limit and not ended_with_eos
    response = {
        "model": model_name,
        "local_raw_output_sha256": hashlib.sha256(raw_answer.encode("utf-8")).hexdigest(),
        "local_raw_output_characters": len(raw_answer),
        "local_raw_output": raw_answer,
        "local_serialized_prompt": prompt,
        "local_effective_generation": effective_generation,
        "local_tokenizer_use_fast": tokenizer_use_fast,
        "choices": [{"message": {"content": answer}, "finish_reason": "length" if reached_limit else "stop"}],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": answer_ids.shape[0],
            "total_tokens": tokens.shape[1],
        },
    }
    return response, "local", parameters


def existing_keys(path: Path, fields: list):
    if not path.exists():
        return set()
    return {tuple(row.get(field) for field in fields) for row in load_jsonl(path)}


def parse_score(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    data = None
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and all(dimension in candidate for dimension in DIMENSIONS):
            data = candidate
            break
    if data is None:
        raise ValueError("No complete six-dimension JSON object was returned")
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


def validate_adapters(candidates):
    for item in candidates:
        adapter_path = item.get("adapter_path")
        if adapter_path:
            adapter = Path(adapter_path)
            if not (adapter / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Adapter configuration not found for {item['label']}: {adapter}")
            if not any((adapter / name).is_file() for name in (
                "adapter_model.safetensors", "adapter_model.bin", "adapter_model.safetensors.index.json"
            )):
                raise FileNotFoundError(f"Adapter weights not found for {item['label']}: {adapter}")


def dry_run(config: dict, questions: list, contexts: dict, candidate_labels, evaluator_labels):
    candidates = enabled_models(config, "candidate_models", candidate_labels)
    evaluators = enabled_models(config, "evaluator_models", evaluator_labels)
    validate_adapters(candidates)
    plan = {
        "run_label": config.get("run_label", ""),
        "question_count": len(questions),
        "question_ids": [row.get("id") for row in questions],
        "retrieval_contexts": {key: {"passages": len(value), "sha256": digest(value)} for key, value in contexts.items()},
        "enabled_candidates": [
            {"label": item["label"], "model": item["model"], "condition": item.get("condition", "prompt_rag"), "backend": item.get("backend", "api"), "api_profile": item.get("api_profile", "default")}
            for item in candidates
        ],
        "enabled_evaluators": [
            {"label": item["label"], "model": item["model"], "api_profile": item.get("api_profile", "default")}
            for item in evaluators
        ],
        "planned_generation_calls": len(questions) * len(candidates),
        "planned_scoring_calls_after_full_generation": len(questions) * len(candidates) * len(evaluators) * int(config.get("review_rounds", 1)),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def generate(config: dict, questions: list, contexts: dict, output_path: Path, sleep_s: float, labels):
    candidates = enabled_models(config, "candidate_models", labels)
    if not candidates:
        raise RuntimeError("No candidate model is enabled")
    validate_adapters(candidates)
    done = existing_keys(output_path, ["run_label", "candidate_label", "question_id"])
    run_label = config.get("run_label", "")
    for model_config in candidates:
        condition = model_config.get("condition", "prompt_rag")
        for question in questions:
            key = (run_label, model_config["label"], question["id"])
            if key in done:
                print(f"skip existing: {model_config['label']} / {question['id']}")
                continue
            passages = contexts.get(question["id"], []) if condition == "prompt_rag" else []
            messages = build_messages(question["question"], passages, condition)
            print(f"generate: {model_config['label']} / {question['id']}")
            started = time.monotonic()
            response, profile_name, parameters = completion(config, model_config, messages, "generation")
            answer = visible_text(response)
            if not answer.strip() or finish_reason(response) == "length":
                append_jsonl(output_path.with_name(output_path.stem + "_failures.jsonl"), {
                    "timestamp_utc": utc_now(),
                    "run_label": run_label,
                    "candidate_label": model_config["label"],
                    "question_id": question["id"],
                    "prompt_sha256": digest(messages),
                    "returned_model": response.get("model", ""),
                    "finish_reason": finish_reason(response),
                    "usage": usage_counts(response),
                    "request_parameters": saved_parameters(parameters),
                    "partial_answer": answer,
                    "response": response,
                })
                raise RuntimeError(f"Empty or truncated answer for {model_config['label']} / {question['id']}; see generation failures log")
            row = {
                "schema_version": 1,
                "record_type": "candidate_answer",
                "timestamp_utc": utc_now(),
                "run_label": run_label,
                "candidate_label": model_config["label"],
                "candidate_model": model_config["model"],
                "condition": condition,
                "api_profile": profile_name,
                "question_id": question["id"],
                "topic": question.get("topic", ""),
                "question": question["question"],
                "retrieved_passages": passages,
                "retrieval_sha256": digest(passages),
                "messages": messages,
                "prompt_sha256": digest(messages),
                "answer": answer,
                "request_parameters": saved_parameters(parameters),
                "finish_reason": finish_reason(response),
                "returned_model": response.get("model", ""),
                "local_raw_output_sha256": response.get("local_raw_output_sha256", ""),
                "local_raw_output_characters": response.get("local_raw_output_characters", 0),
                "local_raw_output": response.get("local_raw_output"),
                "local_serialized_prompt": response.get("local_serialized_prompt"),
                "local_effective_generation": response.get("local_effective_generation"),
                "local_tokenizer_use_fast": response.get("local_tokenizer_use_fast"),
                "usage": usage_counts(response),
                "latency_s": round(time.monotonic() - started, 3),
            }
            append_jsonl(output_path, row)
            time.sleep(sleep_s)
        if model_config.get("backend") == "local":
            import gc
            import torch

            LOCAL_MODELS.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def score(config: dict, answer_path: Path, score_path: Path, sleep_s: float, labels):
    if config.get("scoring_mode", "per_answer") != "per_answer":
        raise ValueError("This configuration requires historical joint scoring; per-answer scoring would change the protocol")
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
            passages = answer_row.get("retrieved_passages") or []
            evidence = "\n\n".join(f"[{i}] {item['source_id']}: {item['text']}" for i, item in enumerate(passages, 1))
            user_prompt = f"问题：\n{answer_row['question']}\n\n检索文献片段：\n{evidence}\n\n待评分回答：\n{answer_row['answer']}"
            messages = [
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            print(f"score: {evaluator['label']} -> {answer_row['candidate_label']} / {answer_row['question_id']}")
            started = time.monotonic()
            response, profile_name, parameters = completion(config, evaluator, messages, "scoring")
            text = visible_text(response)
            if not text.strip() or finish_reason(response) == "length":
                append_jsonl(score_path.with_name(score_path.stem + "_failures.jsonl"), {
                    "timestamp_utc": utc_now(),
                    "run_label": run_label,
                    "evaluator_label": evaluator["label"],
                    "candidate_label": answer_row["candidate_label"],
                    "question_id": answer_row["question_id"],
                    "finish_reason": finish_reason(response),
                    "usage": usage_counts(response),
                    "partial_response": text,
                })
                raise RuntimeError(f"Empty or truncated reviewer score: {evaluator['label']} / {answer_row['candidate_label']} / {answer_row['question_id']}")
            try:
                scores, rationale = parse_score(text)
                parse_error = ""
            except Exception as exc:
                append_jsonl(score_path.with_name(score_path.stem + "_failures.jsonl"), {
                    "timestamp_utc": utc_now(),
                    "run_label": run_label,
                    "evaluator_label": evaluator["label"],
                    "candidate_label": answer_row["candidate_label"],
                    "question_id": answer_row["question_id"],
                    "finish_reason": finish_reason(response),
                    "usage": usage_counts(response),
                    "partial_response": text,
                    "parse_error": f"{type(exc).__name__}: {exc}",
                })
                raise RuntimeError(f"Invalid reviewer score: {evaluator['label']} / {answer_row['candidate_label']} / {answer_row['question_id']}") from exc
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
                "blind_response_id": digest([run_label, answer_row["candidate_label"], answer_row["question_id"]])[:16],
                "messages": messages,
                "scores": scores,
                "rationale": rationale,
                "parse_error": parse_error,
                "request_parameters": saved_parameters(parameters),
                "finish_reason": finish_reason(response),
                "returned_model": response.get("model", ""),
                "local_raw_output_sha256": response.get("local_raw_output_sha256", ""),
                "local_raw_output_characters": response.get("local_raw_output_characters", 0),
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
        description="Generate and score benchmark answers through OpenAI-compatible APIs."
    )
    parser.add_argument("command", choices=["dry-run", "list-models", "generate", "score"])
    parser.add_argument("--config", default="benchmark_config.main.json")
    parser.add_argument("--questions")
    parser.add_argument("--contexts")
    parser.add_argument("--answers")
    parser.add_argument("--scores")
    parser.add_argument("--profile", default="default", help="API profile used by list-models")
    parser.add_argument("--candidate-label", action="append", default=[])
    parser.add_argument("--evaluator-label", action="append", default=[])
    parser.add_argument("--sleep-s", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = resolve(root, args.config)
    config = load_json(config_path)
    args.questions = args.questions or config.get("questions_file", "questions.jsonl")
    args.contexts = args.contexts or config.get("contexts_file", "retrieval_contexts.jsonl")
    args.answers = args.answers or config.get("answers_file", "outputs/model_answers.jsonl")
    args.scores = args.scores or config.get("scores_file", "outputs/model_scores.jsonl")
    if config.get("experiment_type") == "system_comparison":
        for item in enabled_models(config, "candidate_models"):
            role = item.get("component_role")
            expected = "prompt_rag" if role in {"ramad", "rag_ablation"} else "bare"
            if role not in {"ramad", "rag_ablation", "baseline"} or item.get("condition") != expected:
                raise ValueError(f"Invalid system-comparison component assignment: {item['label']}")
    for item in config.get("candidate_models", []):
        if item.get("adapter_path"):
            item["adapter_path"] = str(resolve(config_path.parent, item["adapter_path"]))
    questions_path = resolve(root, args.questions)
    questions = load_jsonl(questions_path) if args.command != "list-models" else []
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    selected = enabled_models(config, "candidate_models", args.candidate_label)
    needs_context = any(item.get("condition", "prompt_rag") == "prompt_rag" for item in selected)
    contexts = load_contexts(resolve(root, args.contexts)) if needs_context and args.command in {"dry-run", "generate"} else {}
    if needs_context and args.command in {"dry-run", "generate"}:
        missing = {row["id"] for row in questions} - set(contexts)
        if missing:
            raise ValueError(f"Missing frozen retrieval contexts: {sorted(missing)}")

    if args.command == "dry-run":
        dry_run(config, questions, contexts, args.candidate_label, args.evaluator_label)
    elif args.command == "list-models":
        list_models(config, args.profile)
    elif args.command == "generate":
        generate(config, questions, contexts, resolve(root, args.answers), args.sleep_s, args.candidate_label)
    elif args.command == "score":
        score(config, resolve(root, args.answers), resolve(root, args.scores), args.sleep_s, args.evaluator_label)


if __name__ == "__main__":
    main()
