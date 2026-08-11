from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Serve the local RAG question-answering interface.')
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
        default="sentence-transformers/all-MiniLM-L6-v2",
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
    parser.add_argument("--allow-pickle", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link.",
    )
    return parser.parse_args()


def make_answer_fn(
    vectorstore: Any,
    tokenizer: Any,
    model: Any,
    top_k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    do_sample: bool,
) -> Any:
    try:
        from .rag_qa import answer_query
    except ImportError:
        from rag_qa import answer_query

    def answer(question: str) -> str:
        if not question or not question.strip():
            return "Please enter a question."
        response, documents = answer_query(
            question=question,
            vectorstore=vectorstore,
            tokenizer=tokenizer,
            model=model,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        sources = []
        for number, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            sources.append(f"[{number}] {source}")
        return f"{response}\n\nSources:\n" + "\n".join(sources)

    return answer


def main() -> None:
    args = parse_args()
    import gradio as gr

    try:
        from .rag_qa import load_model, load_vectorstore
    except ImportError:
        from rag_qa import load_model, load_vectorstore

    vectorstore = load_vectorstore(
        args.index_dir,
        embedding_model_name=args.embedding_model,
        allow_pickle=args.allow_pickle,
    )
    tokenizer, model = load_model(args.model_name_or_path, use_fp16=args.fp16)
    answer_fn = make_answer_fn(
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

    interface = gr.Interface(
        fn=answer_fn,
        inputs=gr.Textbox(label="Question", lines=3),
        outputs=gr.Textbox(label="Answer", lines=16),
        title="Local Raman RAG Interface",
        description="Queries a user-supplied local index and model.",
    )
    interface.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
