from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("training_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one instruction/input prompt through a saved public-candidate LoRA adapter."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-name-or-path",
        help="Local base-model directory or Hugging Face model name. Overrides the config.",
    )
    parser.add_argument(
        "--adapter-path",
        help="Saved adapter directory. Relative paths are resolved from the config directory.",
    )
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--input", default="")
    parser.add_argument("--topic", default="Raman spectroscopy")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int)
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def build_user_content(args: argparse.Namespace) -> str:
    parts = [
        f"Topic: {args.topic.strip()}",
        f"Language: {args.language.strip()}",
        "",
        f"Instruction:\n{args.instruction.strip()}",
    ]
    if args.input.strip():
        parts.extend(["", f"Input:\n{args.input.strip()}"])
    return "\n".join(parts)


def build_prompt(tokenizer: Any, args: argparse.Namespace) -> str:
    user_content = build_user_content(args)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User:\n{user_content}\n\nAssistant:\n"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    config_dir = config_path.parent
    model_name_or_path = args.model_name_or_path or config["model_name_or_path"]
    adapter_path = resolve_path(args.adapter_path or config["output_dir"], config_dir)
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {adapter_path}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict[str, Any] = {}
    if bool(config["fp16"]) and device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        build_prompt(tokenizer, args),
        return_tensors="pt",
    ).to(device)
    max_new_tokens = args.max_new_tokens or int(config["generation"]["max_new_tokens"])
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=bool(config["generation"]["do_sample"]),
            pad_token_id=tokenizer.eos_token_id,
        )
    answer_ids = generated[0][encoded["input_ids"].shape[1] :]
    print(tokenizer.decode(answer_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
