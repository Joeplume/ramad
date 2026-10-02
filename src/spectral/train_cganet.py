"""Train CGANet with the archived source settings and an original training CSV."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cganet_model import RamanMICTransformerFusionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-csv', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError('Choose a new training output directory')
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    frame = pd.read_csv(args.training_csv)
    features = [c for c in frame.columns if c not in ['Category', 'Conc']]
    if len(features) >= 2000:
        features = features[:1800]
    scaler = RobustScaler()
    x = scaler.fit_transform(frame[features].to_numpy()).astype(np.float32)
    encoder = LabelEncoder()
    categories = encoder.fit_transform(frame['Category'])
    concentrations = frame['Conc'].to_numpy(dtype=np.float32)
    train_val, test = train_test_split(np.arange(len(frame)), test_size=0.2, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=42)

    def loader(rows, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(x[rows]),
                          torch.tensor(categories[rows], dtype=torch.long),
                          torch.from_numpy(concentrations[rows])), batch_size=32, shuffle=shuffle)

    train_loader, val_loader = loader(train, True), loader(val, False)
    model = RamanMICTransformerFusionModel(input_dim=len(features), num_categories=len(encoder.classes_))

    def initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    model.apply(initialize)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    category_loss, concentration_loss = nn.CrossEntropyLoss(), nn.MSELoss()
    args.out_dir.mkdir(parents=True)
    membership = np.full(len(frame), 'test', dtype='<U5')
    membership[train], membership[val] = 'train', 'val'
    pd.DataFrame({'source_row': np.arange(len(frame)), 'split': membership}).to_csv(args.out_dir / 'split.csv', index=False)
    np.savez_compressed(args.out_dir / 'preprocessing.npz', center=scaler.center_, scale=scaler.scale_,
                        feature_names=np.asarray(features), category_names=encoder.classes_.astype(str))
    (args.out_dir / 'run.json').write_text(json.dumps({'seed': 42, 'epochs': 300,
        'training_csv': str(args.training_csv), 'rows': len(frame), 'features': len(features),
        'train_rows': len(train), 'validation_rows': len(val), 'test_rows': len(test),
        'device': str(device), 'torch': torch.__version__}, indent=2) + '\n', encoding='utf-8')
    best_loss = float('inf')
    history = []
    for epoch in range(300):
        model.train()
        train_total, skipped = 0.0, 0
        for bx, by, bc in train_loader:
            if torch.isnan(bx).any() or torch.isinf(bx).any():
                bx = torch.nan_to_num(bx, nan=0.0, posinf=1.0, neginf=0.0)
            bx, by, bc = bx.to(device), by.to(device), bc.to(device)
            optimizer.zero_grad()
            logits, values = model(bx)
            if torch.isnan(logits).any() or torch.isnan(values).any():
                skipped += 1
                continue
            lc, lr = category_loss(logits, by), concentration_loss(values.flatten(), bc)
            if torch.isnan(lc) or torch.isnan(lr):
                skipped += 1
                continue
            loss = lc + lr
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_total += loss.item()
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for bx, by, bc in val_loader:
                logits, values = model(bx.to(device))
                val_total += (category_loss(logits, by.to(device)) +
                              concentration_loss(values.flatten(), bc.to(device))).item()
        val_mean = val_total / len(val_loader)
        scheduler.step(val_mean)
        if val_mean < best_loss:
            best_loss = val_mean
            torch.save(model.state_dict(), args.out_dir / 'checkpoint.pth')
        row = {'epoch': epoch + 1, 'training_loss': train_total / len(train_loader),
               'validation_loss': val_mean, 'learning_rate': optimizer.param_groups[0]['lr'],
               'skipped_training_batches': skipped}
        history.append(row)
        pd.DataFrame(history).to_csv(args.out_dir / 'training_history.csv', index=False)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
