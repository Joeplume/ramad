"""Generate the Kimi bare-condition answers with SSE streaming.

The model, prompts, decoding parameters, and saved record schema match
run_benchmark.py; only the HTTP transfer mode differs to avoid idle proxy drops.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))
import run_benchmark as rb  # noqa: E402


def stream_completion(config, model_config, messages):
    profile = rb.api_profile(config, model_config)
    parameters = dict(config["generation"])
    parameters.update(model_config.get("request_overrides", {}))
    payload = {"model": model_config["model"], "messages": messages, **parameters, "stream": True, "stream_options": {"include_usage": True}}
    api_key = os.environ[profile["api_key_env"]]
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream", "Authorization": f"Bearer {api_key}"}
    url = profile["base_url"].rstrip("/") + profile.get("chat_completions_path", "/v1/chat/completions")
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), method="POST", headers=headers)
    with urllib.request.urlopen(request, timeout=min(int(profile.get("request_timeout_s", 600)), 120)) as response:
        parts = []
        returned_model = ""
        reason = ""
        usage = {}
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            returned_model = chunk.get("model") or returned_model
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
                if len(parts) % 50 == 0:
                    print(".", end="", flush=True)
            reason = choice.get("finish_reason") or reason
    return "".join(parts), returned_model, reason or "stop", usage, rb.saved_parameters(parameters), profile["name"]


def main():
    config_path = Path(__file__).with_name("benchmark_config.json")
    config = rb.load_json(config_path)
    questions = rb.load_jsonl(BENCHMARK_DIR / config["questions_file"])
    model_config = next(item for item in config["candidate_models"] if item["label"] == "Kimi-K3-bare")
    output_path = BENCHMARK_DIR / config["answers_file"]
    done = rb.existing_keys(output_path, ["run_label", "candidate_label", "question_id"])
    for question in questions:
        key = (config["run_label"], model_config["label"], question["id"])
        if key in done:
            continue
        messages = rb.build_messages(question["question"], [], "bare")
        for attempt in range(8):
            try:
                print(f"stream generate: {question['id']} attempt={attempt + 1}", flush=True)
                started = time.monotonic()
                answer, returned_model, reason, usage, parameters, profile_name = stream_completion(config, model_config, messages)
                if not answer.strip() or reason == "length":
                    raise RuntimeError(f"Incomplete streamed answer: finish_reason={reason!r}, characters={len(answer)}")
                row = {
                    "schema_version": 1,
                    "record_type": "candidate_answer",
                    "timestamp_utc": rb.utc_now(),
                    "run_label": config["run_label"],
                    "candidate_label": model_config["label"],
                    "candidate_model": model_config["model"],
                    "condition": "bare",
                    "api_profile": profile_name,
                    "question_id": question["id"],
                    "topic": question.get("topic", ""),
                    "question": question["question"],
                    "retrieved_passages": [],
                    "retrieval_sha256": rb.digest([]),
                    "messages": messages,
                    "prompt_sha256": rb.digest(messages),
                    "answer": answer,
                    "request_parameters": parameters,
                    "finish_reason": reason,
                    "returned_model": returned_model,
                    "local_raw_output_sha256": "",
                    "local_raw_output_characters": 0,
                    "local_raw_output": None,
                    "local_serialized_prompt": None,
                    "local_effective_generation": None,
                    "local_tokenizer_use_fast": None,
                    "usage": {key: value for key, value in usage.items() if key in {"prompt_tokens", "completion_tokens", "total_tokens"}},
                    "latency_s": round(time.monotonic() - started, 3),
                    "transport": "sse_stream",
                }
                rb.append_jsonl(output_path, row)
                print(f" completed characters={len(answer)}")
                break
            except Exception as exc:
                if attempt == 7:
                    raise
                wait = 5 * (attempt + 1)
                print(f"stream attempt failed: {type(exc).__name__}; retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
        time.sleep(2)


if __name__ == "__main__":
    main()
