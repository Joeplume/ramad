import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


FILES = (
    "augmented_MG_3870_02181807.csv",
    "augmented_OFX_3871_02181811.csv",
    "augmented_STZ_3844_02181809.csv",
)
FEATURES = [str(index) for index in range(1600)]


def main():
    parser = argparse.ArgumentParser(description="Recreate the mixed-drug CGANet scaler and saved row split.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frames = []
    identity = []
    for name in FILES:
        frame = pd.read_csv(args.data_dir / name, usecols=["Category", "Conc", *FEATURES]).fillna(0)
        if set(FEATURES) - set(frame.columns):
            raise ValueError(f"Missing spectral channels in {name}")
        frames.append(frame)
        identity.extend((name, index) for index in range(len(frame)))
    combined = pd.concat(frames, ignore_index=True)
    x = combined[FEATURES].to_numpy(dtype=np.float64)
    labels = LabelEncoder()
    y = labels.fit_transform(combined["Category"])
    scaler = RobustScaler().fit(x)
    indices = np.arange(len(combined))
    train, test = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    partition = {int(index): "train" for index in train}
    partition.update({int(index): "internal_test" for index in test})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "mixed_1600_preprocessing.npz",
        center=scaler.center_,
        scale=scaler.scale_,
        feature_names=np.array(FEATURES, dtype="U8"),
        category_names=np.array(labels.classes_, dtype="U64"),
    )
    with (args.out_dir / "mixed_1600_split.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["combined_index", "source_file", "source_row_zero_based", "partition"])
        for index, (source_file, source_row) in enumerate(identity):
            writer.writerow([index, source_file, source_row, partition[index]])
    (args.out_dir / "mixed_1600_data_info.json").write_text(
        json.dumps({"input_channels": 1600, "category_names": labels.classes_.tolist(),
                    "train_rows": len(train), "internal_test_rows": len(test),
                    "external_test_file": "test_data_250218.csv", "random_seed": 42}, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(train)} training rows and {len(test)} internal test rows")


if __name__ == "__main__":
    main()
