from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from langchain_core.documents import Document
from pdf2image import convert_from_path


def extract_text_from_pdf(
    pdf_path: str | Path,
    use_ocr: bool = False,
    poppler_path: str | Path | None = None,
    tesseract_cmd: str | Path | None = None,
) -> str:
    with fitz.open(str(pdf_path)) as pdf:
        text = "".join(page.get_text() for page in pdf)
    if text.strip() or not use_ocr:
        return text
    return ocr_pdf(pdf_path, poppler_path=poppler_path, tesseract_cmd=tesseract_cmd)


def ocr_pdf(
    pdf_path: str | Path,
    poppler_path: str | Path | None = None,
    tesseract_cmd: str | Path | None = None,
) -> str:
    if tesseract_cmd is not None:
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
    images = convert_from_path(
        str(pdf_path),
        dpi=300,
        poppler_path=str(poppler_path) if poppler_path is not None else None,
    )
    return "\n".join(
        pytesseract.image_to_string(image, lang="eng+chi_sim") for image in images
    )


def extract_metadata(pdf_path: str | Path) -> dict[str, str]:
    path = Path(pdf_path)
    return {"source": path.stem, "source_file": path.name}


def clean_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build_structured_chunks(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    chunking_strategy: str = "default",
    combine_text_under_n_chars: int = 50,
) -> list[Document]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [item.strip() for item in text.split("\n") if item.strip()]
    if chunking_strategy == "paragraph":
        return [
            Document(page_content=paragraph, metadata={**metadata, "paragraph": index})
            for index, paragraph in enumerate(paragraphs)
            if len(paragraph) >= combine_text_under_n_chars
        ]

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "。", "!", "！", "?", "？", ",", "，"],
    )
    return [
        Document(page_content=chunk, metadata=metadata)
        for chunk in splitter.split_text(text)
    ]
