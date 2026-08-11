from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("training_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter from instruction/input/output JSONL records."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-name-or-path",
        help="Local model directory or Hugging Face model name. Overrides the config.",
    )
    parser.add_argument(
        "--data-path",
        help="JSONL path. Relative paths are resolved from the config directory.",
    )
    parser.add_argument(
        "--output-dir",
        help="Adapter output directory. Relative paths are resolved from the config directory.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()


def require_fields(record: dict[str, Any], required_fields: list[str]) -> None:
    missing = [field for field in required_fields if field not in record]
    if missing:
        raise ValueError(f"JSONL record is missing required fields: {', '.join(missing)}")
    if not str(record["instruction"]).strip() or not str(record["output"]).strip():
        raise ValueError("Each JSONL record needs non-empty instruction and output values.")


def build_user_content(record: dict[str, Any]) -> str:
    parts = [
        f"Topic: {str(record['topic']).strip()}",
        f"Language: {str(record['language']).strip()}",
        "",
        f"Instruction:\n{str(record['instruction']).strip()}",
    ]
    user_input = str(record["input"]).strip()
    if user_input:
        parts.extend(["", f"Input:\n{user_input}"])
    return "\n".join(parts)


def build_prompt(tokenizer: Any, record: dict[str, Any]) -> str:
    user_content = build_user_content(record)
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

    data_path = resolve_path(args.data_path or config["data_path"], config_dir)
    output_dir = resolve_path(args.output_dir or config["output_dir"], config_dir)
    model_name_or_path = args.model_name_or_path or config["model_name_or_path"]
    if not data_path.is_file():
        raise FileNotFoundError(f"Instruction JSONL not found: {data_path}")

    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    use_fp16 = bool(config["fp16"])
    if use_fp16 and not torch.cuda.is_available():
        raise RuntimeError("fp16 is configured but no CUDA device is available.")

    raw_records = load_dataset("json", data_files=str(data_path), split="train")
    if len(raw_records) < 2:
        raise ValueError("At least two JSONL records are required for the configured 95/5 split.")

    required_fields = list(config["required_fields"])
    for record in raw_records:
        require_fields(record, required_fields)

    holdout_size = max(1, round(len(raw_records) * (1 - float(config["train_fraction"]))))
    partitions = raw_records.train_test_split(
        test_size=holdout_size,
        seed=int(config["seed"]),
        shuffle=True,
    )
    train_records = partitions["train"]
    holdout_records = partitions["test"]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {}
    if use_fp16:
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=list(config["lora"]["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    max_input_length = int(config["max_input_length"])
    max_output_length = int(config["max_output_length"])

    def tokenize_record(record: dict[str, Any]) -> dict[str, list[int]]:
        prompt_ids = tokenizer(
            build_prompt(tokenizer, record),
            add_special_tokens=False,
            truncation=True,
            max_length=max_input_length,
        )["input_ids"]
        answer_ids = tokenizer(
            str(record["output"]).strip() + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=max_output_length,
        )["input_ids"]
        return {
            "input_ids": prompt_ids + answer_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
            "labels": [-100] * len(prompt_ids) + answer_ids,
        }

    train_tokens = train_records.map(
        tokenize_record,
        remove_columns=train_records.column_names,
        desc="Tokenizing training records",
    )
    holdout_tokens = holdout_records.map(
        tokenize_record,
        remove_columns=holdout_records.column_names,
        desc="Tokenizing holdout records",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(config["num_train_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "optim": str(config["optimizer"]),
        "fp16": use_fp16,
        "logging_steps": int(config["logging_steps"]),
        "save_strategy": "epoch",
        "save_total_limit": int(config["save_total_limit"]),
        "report_to": [],
        "seed": int(config["seed"]),
    }
    argument_parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in argument_parameters:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=holdout_tokens,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            padding=True,
        ),
    )
    print(f"Training records: {len(train_records)}; holdout records: {len(holdout_records)}")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"LoRA adapter and tokenizer written to: {output_dir}")


if __name__ == "__main__":
    main()
