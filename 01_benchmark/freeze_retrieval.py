import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Freeze one retrieved evidence set per benchmark question.")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, default=Path(__file__).with_name("questions.jsonl"))
    parser.add_argument("--doi-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("retrieval_contexts.jsonl"))
    parser.add_argument("--embedding-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-pickle", action="store_true")
    args = parser.parse_args()
    if not args.allow_pickle:
        parser.error("--allow-pickle is required for a trusted local FAISS index")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    with args.doi_map.open(encoding="utf-8-sig", newline="") as handle:
        mapping = {row["source_file"].strip(): row for row in csv.DictReader(handle)}
    if not mapping:
        raise ValueError("DOI mapping is empty")
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    index = FAISS.load_local(
        str(args.index_dir), embeddings, allow_dangerous_deserialization=True
    )
    questions = load_jsonl(args.questions)
    if len({row["id"] for row in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    rows = []
    for question in questions:
        matches = index.similarity_search_with_score(question["question"], k=args.top_k)
        passages = []
        for document, distance in matches:
            source_file = str(document.metadata.get("source_file", "")).strip()
            source = mapping.get(source_file)
            if source is None or not str(source.get("doi", "")).strip():
                raise ValueError(f"DOI mapping missing for {source_file or 'unknown source'}")
            passages.append({
                "source_id": source_file,
                "doi": source["doi"].strip(),
                "page": document.metadata.get("page", ""),
                "title": source.get("title", "").strip(),
                "distance": float(distance),
                "text": document.page_content.strip(),
            })
        passages.sort(key=lambda item: (item["distance"], item["source_id"], str(item["page"])))
        if len(passages) != args.top_k:
            raise ValueError(f"Expected {args.top_k} passages for {question['id']}, found {len(passages)}")
        rows.append({"question_id": question["id"], "passages": passages})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Frozen {len(rows)} question contexts in {args.out}")


if __name__ == "__main__":
    main()
