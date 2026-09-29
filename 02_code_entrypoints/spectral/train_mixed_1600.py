import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from cganet_model import RamanMICTransformerFusionModel


WEIGHT_RANGES = {
    "MG": (10.6, 11.0),
    "STZ": (8.6, 9.0),
    "OFX": (9.6, 10.0),
}


def sample_weight(category, concentration):
    low, high = WEIGHT_RANGES[category]
    if concentration > high:
        return 0.1
    if concentration < low:
        return 1.0
    return max((1.0 - (concentration - low) / (high - low)) ** 2, 0.1)


def load_data(package_dir):
    metadata = np.load(package_dir / "mixed_1600_preprocessing.npz", allow_pickle=False)
    features = metadata["feature_names"].tolist()
    categories = metadata["category_names"].tolist()
    with (package_dir / "mixed_1600_split.csv").open(encoding="utf-8", newline="") as handle:
        split = list(csv.DictReader(handle))
    filenames = list(dict.fromkeys(row["source_file"] for row in split))
    frames = [pd.read_csv(package_dir / name, usecols=["Category", "Conc", *features]).fillna(0) for name in filenames]
    frame = pd.concat(frames, ignore_index=True)
    if len(frame) != len(split):
        raise ValueError("Saved split and data row counts differ")
    for index, row in enumerate(split):
        if int(row["combined_index"]) != index:
            raise ValueError(f"Invalid combined row index at {index}")
    x = frame[features].to_numpy(dtype=np.float64)
    x = ((x - metadata["center"]) / metadata["scale"]).astype(np.float32)
    y_category = np.array([categories.index(value) for value in frame["Category"].astype(str)], dtype=np.int64)
    y_concentration = frame["Conc"].to_numpy(dtype=np.float32)
    partitions = {name: np.array([i for i, row in enumerate(split) if row["partition"] == name], dtype=np.int64)
                  for name in ("train", "internal_test")}
    if not len(partitions["train"]) or not len(partitions["internal_test"]):
        raise ValueError("The saved split needs training and internal test rows")
    return x, y_category, y_concentration, categories, partitions


def main():
    parser = argparse.ArgumentParser(description="Retrain the released mixed-drug CGANet architecture.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size must be at least two")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    x, y_category, y_concentration, categories, partitions = load_data(args.package_dir)
    data = {}
    for name, indices in partitions.items():
        data[name] = TensorDataset(
            torch.from_numpy(x[indices]),
            torch.from_numpy(y_category[indices]),
            torch.from_numpy(y_concentration[indices]),
        )
    train_indices = partitions["train"]
    weights = torch.tensor(
        [sample_weight(categories[int(y_category[index])], float(y_concentration[index])) for index in train_indices],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, sampler=sampler)
    test_loader = DataLoader(data["internal_test"], batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RamanMICTransformerFusionModel(input_dim=1600, num_categories=len(categories)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    classification_loss = torch.nn.CrossEntropyLoss()
    concentration_loss = torch.nn.MSELoss()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        training = []
        for spectra, category, concentration in train_loader:
            spectra, category, concentration = spectra.to(device), category.to(device), concentration.to(device)
            optimizer.zero_grad()
            logits, predicted = model(spectra)
            loss = classification_loss(logits, category) + concentration_loss(predicted.squeeze(1), concentration)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            training.append(float(loss.item()))
        model.eval()
        validation = []
        with torch.inference_mode():
            for spectra, category, concentration in test_loader:
                spectra, category, concentration = spectra.to(device), category.to(device), concentration.to(device)
                logits, predicted = model(spectra)
                loss = classification_loss(logits, category) + concentration_loss(predicted.squeeze(1), concentration)
                validation.append(float(loss.item()))
        result = {"epoch": epoch, "train_loss": sum(training) / len(training),
                  "internal_test_loss": sum(validation) / len(validation)}
        history.append(result)
        if result["internal_test_loss"] < best:
            best = result["internal_test_loss"]
            torch.save(model.state_dict(), args.out_dir / "best_model.pth")
        print(f"epoch={epoch} train={result['train_loss']:.6f} internal_test={result['internal_test_loss']:.6f}")
    with (args.out_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "internal_test_loss"])
        writer.writeheader()
        writer.writerows(history)
    (args.out_dir / "training_run.json").write_text(
        json.dumps({"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                    "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-5,
                    "sample_weights": WEIGHT_RANGES, "train_rows": len(train_indices),
                    "internal_test_rows": len(partitions["internal_test"]), "device": str(device)}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
