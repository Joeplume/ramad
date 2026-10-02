"""Evaluate the provenance-linked internal test cohort with unchanged labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

from cganet_model import RamanMICTransformerFusionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError('Choose a new result directory')
    torch.set_num_threads(4)
    data = np.load(args.package_dir / 'internal_test.npz', allow_pickle=False)
    preprocessing = np.load(args.package_dir / 'preprocessing.npz', allow_pickle=False)
    names = preprocessing['category_names'].tolist()
    x = ((data['spectra'] - preprocessing['center']) / preprocessing['scale']).astype(np.float32)
    x = np.nan_to_num(x, nan=0, posinf=1, neginf=0)
    observed_category = data['categories']
    y = data['concentrations']
    model = RamanMICTransformerFusionModel(input_dim=x.shape[1], num_categories=len(names))
    model.load_state_dict(torch.load(args.package_dir / 'checkpoint.pth', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    predicted_category, predicted_concentration = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            logits, values = model(torch.from_numpy(x[start:start + 32]))
            predicted_category.extend(names[i] for i in logits.argmax(1).tolist())
            predicted_concentration.extend(values.flatten().tolist())
    pred = np.asarray(predicted_concentration, dtype=np.float32)
    metrics = {'samples': len(y), 'accuracy': float(accuracy_score(observed_category, predicted_category)),
               'rmse': float(np.sqrt(mean_squared_error(y, pred))), 'r2': float(r2_score(y, pred)),
               'category_order': names,
               'confusion_matrix': confusion_matrix(observed_category, predicted_category, labels=names).tolist()}
    args.out_dir.mkdir(parents=True)
    pd.DataFrame({'source_row': data['source_rows'], 'category': observed_category,
                  'predicted_category': predicted_category, 'concentration': y,
                  'predicted_concentration': pred}).to_csv(args.out_dir / 'predictions.csv', index=False)
    (args.out_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
