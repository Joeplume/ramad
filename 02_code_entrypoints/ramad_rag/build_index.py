from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a local FAISS index from a user-supplied PDF directory.')
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/pdfs"),
        help="Relative or absolute directory containing PDF files.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/faiss_index"),
        help="Relative or absolute output directory for the FAISS index.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformers model used for indexing.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use OCR when a PDF has no extractable text.",
    )
    parser.add_argument(
        "--poppler-path",
        type=Path,
        help="Optional Poppler bin directory used only for OCR.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=Path,
        help="Optional path to the Tesseract executable used only for OCR.",
    )
    return parser.parse_args()


def build_index(
    pdf_dir: Path,
    index_dir: Path,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    use_ocr: bool = False,
    poppler_path: Path | None = None,
    tesseract_cmd: Path | None = None,
) -> None:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from transformers import AutoTokenizer

    try:
        from .utils_pdf import build_structured_chunks, extract_metadata, extract_pages_from_pdf
    except ImportError:
        from utils_pdf import build_structured_chunks, extract_metadata, extract_pages_from_pdf

    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size.")

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_fast=True)
    token_length = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    documents = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            pages = extract_pages_from_pdf(
                pdf_path,
                use_ocr=use_ocr,
                poppler_path=poppler_path,
                tesseract_cmd=tesseract_cmd,
            )
            for page_number, text in pages:
                documents.extend(
                    build_structured_chunks(
                        text,
                        {**extract_metadata(pdf_path), "page": page_number},
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=token_length,
                    )
                )
        except Exception as exc:
            print(f"Skipping {pdf_path.name}: {exc}")

    if not documents:
        raise RuntimeError("No indexable text was produced from the supplied PDFs.")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Index written to: {index_dir}")


def main() -> None:
    args = parse_args()
    build_index(
        pdf_dir=args.pdf_dir,
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_ocr=args.ocr,
        poppler_path=args.poppler_path,
        tesseract_cmd=args.tesseract_cmd,
    )


if __name__ == "__main__":
    main()
