import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_squared_error

from cganet_model import RamanMICTransformerFusionModel


def main():
    parser = argparse.ArgumentParser(description="Evaluate a released mixed-drug CGANet checkpoint.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    data_path = args.package_dir / "test_data_250218.csv"
    state_path = args.package_dir / "cganet_mixed_1600.pth"
    metadata_path = args.package_dir / "mixed_1600_preprocessing.npz"
    metadata = np.load(metadata_path, allow_pickle=False)
    features = metadata["feature_names"].tolist()
    categories = metadata["category_names"].tolist()
    frame = pd.read_csv(data_path, usecols=["Category", "Conc", *features]).fillna(0)
    x = frame[features].to_numpy(dtype=np.float64)
    x = (x - metadata["center"]) / metadata["scale"]
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model = RamanMICTransformerFusionModel(input_dim=len(features), num_categories=len(categories))
    model.load_state_dict(state, strict=True)
    model.eval()
    predicted_category = []
    predicted_concentration = []
    with torch.inference_mode():
        for start in range(0, len(x), args.batch_size):
            category_logits, concentration = model(torch.from_numpy(x[start:start + args.batch_size]))
            predicted_category.extend(categories[index] for index in category_logits.argmax(dim=1).tolist())
            predicted_concentration.extend(concentration.squeeze(1).tolist())
    observed_category = frame["Category"].astype(str).tolist()
    observed_concentration = frame["Conc"].to_numpy(dtype=float)
    results = {
        "samples": len(frame),
        "classification_accuracy": float(accuracy_score(observed_category, predicted_category)),
        "concentration_rmse": float(mean_squared_error(observed_concentration, predicted_concentration) ** 0.5),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "external_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_zero_based", "observed_category", "predicted_category", "observed_concentration", "predicted_concentration"])
        for index, row in enumerate(zip(observed_category, predicted_category, observed_concentration, predicted_concentration)):
            writer.writerow([index, *row])
    (args.out_dir / "external_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
