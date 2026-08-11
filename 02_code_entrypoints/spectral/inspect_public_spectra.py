from csv import DictReader
from pathlib import Path


DATA_PATH = Path(__file__).with_name("external_test_spectra_two_examples.csv")


def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    spectrum_columns = [name for name in fieldnames if name not in {"Category", "Conc"}]
    print(f"data file: {DATA_PATH.name}")
    print(f"records: {len(rows)}")
    print(f"columns: {len(fieldnames)}")
    print(f"spectrum columns: {len(spectrum_columns)}")
    print("labels:")
    for row in rows:
        print(f"  Category={row['Category']}; Conc={row['Conc']}")


if __name__ == "__main__":
    main()
