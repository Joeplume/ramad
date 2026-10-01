from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = (
    "id",
    "instruction",
    "input",
    "output",
    "topic",
    "language",
)


def inspect(path: Path) -> int:
    topics: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    ids: list[str] = []
    missing: list[tuple[int, list[str]]] = []
    malformed: list[int] = []
    record_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line_number)
                continue

            record_count += 1
            ids.append(str(row.get("id", "")))
            topics[str(row.get("topic", ""))] += 1
            languages[str(row.get("language", ""))] += 1
            absent = [field for field in REQUIRED_FIELDS if not row.get(field)]
            if absent:
                missing.append((line_number, absent))

    duplicate_ids = sorted(
        item for item, count in Counter(ids).items() if item and count > 1
    )

    print(f"records: {record_count}")
    print("topics:")
    for topic, count in sorted(topics.items()):
        print(f"  {topic}: {count}")
    print("languages:")
    for language, count in sorted(languages.items()):
        print(f"  {language}: {count}")
    print(f"missing_required_fields: {len(missing)}")
    for line_number, absent in missing:
        print(f"  line {line_number}: {', '.join(absent)}")
    print(f"duplicate_ids: {len(duplicate_ids)}")
    for item in duplicate_ids:
        print(f"  {item}")
    print(f"malformed_lines: {len(malformed)}")
    for line_number in malformed:
        print(f"  line {line_number}")

    return 1 if missing or malformed or duplicate_ids else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl",
        nargs="?",
        default=str(Path(__file__).with_name("representative_training_examples_100.jsonl")),
        help="JSONL file to inspect",
    )
    args = parser.parse_args()
    return inspect(Path(args.jsonl))


if __name__ == "__main__":
    raise SystemExit(main())

