"""Evaluate the archived 1800-channel checkpoint with explicit label provenance."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--selection', choices=['all', 'archived-half'], default='all')
    args = parser.parse_args()
    package = args.package_dir.resolve()
    config = json.loads((package / 'model_config.json').read_text(encoding='utf-8'))
    if sha256(package / 'external.csv') != config['external_sha256']:
        raise ValueError('External table differs from the table covered by the stored label mapping')
    frame = pd.read_csv(package / 'external.csv')
    labels = pd.read_csv(package / 'external_label_map.csv')
    if labels['row_zero_based'].tolist() != list(range(len(frame))):
        raise ValueError('Label map must cover every external row exactly once and in order')
    if labels.stored_category.tolist() != frame.Category.tolist() or not np.array_equal(labels.stored_conc, frame.Conc):
        raise ValueError('Label map does not match stored labels')
    expected_labels = np.where(frame.Conc.to_numpy() == 10, 'Water', frame.Category.to_numpy())
    if not np.array_equal(labels.canonical_category.to_numpy(), expected_labels):
        raise ValueError('Label map differs from the archived augmentation rule for this table')
    chosen = frame.sample(frac=0.5, random_state=42).index.to_numpy() if args.selection == 'archived-half' else np.arange(len(frame))
    metadata = np.load(package / 'preprocessing.npz', allow_pickle=False)
    features, categories = metadata['feature_names'].tolist(), metadata['category_names'].tolist()
    x = frame.iloc[chosen][features].to_numpy(dtype=np.float64)
    x = ((x - metadata['center']) / metadata['scale']).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite external features')
    spec = importlib.util.spec_from_file_location('archived_1800_model', package / 'archived_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.RamanMICTransformerFusionModel(input_dim=len(features), num_categories=len(categories))
    model.load_state_dict(torch.load(package / 'checkpoint.pth', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    torch.set_num_threads(4)
    predicted, concentrations = [], []
    with torch.inference_mode():
        for start in range(0, len(x), 32):
            category, concentration = model(torch.from_numpy(x[start:start + 32]))
            predicted.extend(categories[i] for i in category.argmax(1).tolist())
            concentrations.extend(concentration.flatten().tolist())
    result = labels.iloc[chosen].copy()
    result['predicted_category'] = predicted
    result['predicted_concentration'] = concentrations
    stats = {'selection': args.selection, 'samples': len(result),
             'accuracy_stored_labels': float(np.mean(result.stored_category == predicted)),
             'accuracy_source_water_labels': float(np.mean(result.canonical_category == predicted)),
             'mapped_control_rows': int(np.sum(result.stored_category != result.canonical_category)),
             'concentration_rmse': float(np.sqrt(np.mean((result.stored_conc - concentrations) ** 2))),
             'label_rule_scope': 'Only this hashed archived external table; Conc=10 is its Water control code.'}
    if args.out_dir.exists():
        raise ValueError('Choose a new output directory to preserve earlier predictions')
    args.out_dir.mkdir(parents=True)
    result.to_csv(args.out_dir / 'predictions.csv', index=False)
    (args.out_dir / 'metrics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
