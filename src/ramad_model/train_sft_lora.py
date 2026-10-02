from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).with_name("training_config_qlora.json")


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


def load_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate record IDs in {path}")
    return ids


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
    model_revision = config.get("model_revision") if not args.model_name_or_path else None
    if not data_path.is_file():
        raise FileNotFoundError(f"Instruction JSONL not found: {data_path}")

    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
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

    train_ids = load_ids(resolve_path(config["train_ids_path"], config_dir))
    validation_ids = load_ids(resolve_path(config["validation_ids_path"], config_dir))
    dataset_ids = [str(record["id"]) for record in raw_records]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Training corpus contains duplicate record IDs")
    if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != set(dataset_ids):
        raise ValueError("Saved train/validation split does not partition the training corpus")
    position = {record_id: index for index, record_id in enumerate(dataset_ids)}
    train_records = raw_records.select([position[record_id] for record_id in train_ids])
    holdout_records = raw_records.select([position[record_id] for record_id in validation_ids])

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=model_revision, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {}
    if use_fp16:
        model_kwargs["torch_dtype"] = torch.float16
    if config.get("load_in_4bit", False):
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit training requires a CUDA device")
        model_kwargs["device_map"] = {"": 0}
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if use_fp16 else torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision=model_revision, **model_kwargs)
    model.config.use_cache = False
    if config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        )

    target_modules = list(config["lora"]["target_modules"])
    upper_layer_count = config["lora"].get("upper_layer_count")
    if upper_layer_count is not None:
        layer_count = int(model.config.num_hidden_layers)
        if not 0 < int(upper_layer_count) <= layer_count:
            raise ValueError("upper_layer_count must be between 1 and the model's layer count")
        start = layer_count - int(upper_layer_count)
        target_modules = [
            f"model.layers.{layer}.self_attn.{projection}"
            for layer in range(start, layer_count)
            for projection in target_modules
        ]
        if config["lora"].get("include_lm_head", False):
            target_modules.append("lm_head")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora"]["r"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=target_modules,
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
    manifest = {
        "base_model": model_name_or_path,
        "base_model_revision": model_revision,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "seed": int(config["seed"]),
        "config": config,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "gradient_checkpointing": bool(config.get("gradient_checkpointing", False)),
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
