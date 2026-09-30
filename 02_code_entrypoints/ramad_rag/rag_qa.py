from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
SYSTEM_PROMPT = (PROMPT_DIR / "domain_system.txt").read_text(encoding="utf-8").strip()
RAG_USER_TEMPLATE = (PROMPT_DIR / "rag_user.txt").read_text(encoding="utf-8").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Query a local FAISS index with a local causal language model.')
    parser.add_argument("--question", required=True, help="Question to answer.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/faiss_index"),
        help="Relative or absolute FAISS index directory.",
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Hugging Face model identifier or local model directory.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Must match the model used to build the index.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample", dest="do_sample", action="store_true")
    sampling.add_argument("--greedy", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--allow-pickle",
        action="store_true",
        help="Required to load the local FAISS metadata file.",
    )
    parser.add_argument("--show-sources", action="store_true")
    return parser.parse_args()


def load_vectorstore(
    index_dir: Path,
    embedding_model_name: str,
    allow_pickle: bool,
) -> Any:
    if not allow_pickle:
        raise ValueError(
            "Loading the local FAISS metadata file requires the explicit --allow-pickle flag."
        )

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_model(model_name_or_path: str, use_fp16: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict[str, Any] = {}
    if use_fp16 and device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def answer_query(
    question: str,
    vectorstore: Any,
    tokenizer: Any,
    model: Any,
    top_k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> tuple[str, list[Any]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    documents = vectorstore.similarity_search(question, k=top_k)
    context = "\n\n".join(
        f"[{index}] source_id={document.metadata.get('source_file', document.metadata.get('source', 'unknown'))} page={document.metadata.get('page', '')}\n{document.page_content}"
        for index, document in enumerate(documents, start=1)
    )
    user_prompt = RAG_USER_TEMPLATE.format(context=context, question=question)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) if getattr(tokenizer, "chat_template", None) else f"system:\n{SYSTEM_PROMPT}\n\nuser:\n{user_prompt}\n\nassistant:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": temperature, "top_p": top_p})

    output_ids = model.generate(**inputs, **generation_kwargs)
    answer_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer, documents


def main() -> None:
    args = parse_args()
    vectorstore = load_vectorstore(
        args.index_dir,
        embedding_model_name=args.embedding_model,
        allow_pickle=args.allow_pickle,
    )
    tokenizer, model = load_model(args.model_name_or_path, use_fp16=args.fp16)
    answer, documents = answer_query(
        question=args.question,
        vectorstore=vectorstore,
        tokenizer=tokenizer,
        model=model,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
    )
    print(answer)
    if args.show_sources:
        for number, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            print(f"\n[{number}] {source}")


if __name__ == "__main__":
    main()
