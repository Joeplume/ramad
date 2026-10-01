"""Apply the recovered aggregation methods to a complete fresh joint review."""
import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from reproduce_historical_scores import DIMENSIONS, REVIEWERS, aggregate, read_csv, write_csv


def summarize(rows):
    aliases = {'DeepSeek-V3': 'Deepseek-V3', 'Qwen3-4B': 'Qwen3-4b'}
    rows = [{**r, 'reviewer': aliases.get(r['reviewer'], r['reviewer']),
             'candidate': aliases.get(r['candidate'], r['candidate'])} for r in rows]
    candidates = {r['candidate'] for r in rows}
    if len(candidates) != 6 or not set(REVIEWERS).issubset(candidates):
        raise ValueError('Expected the six-candidate comparison, including the four reviewer models')
    expected = {(r, c, n) for r in REVIEWERS for c in candidates for n in (1, 2, 3)}
    counts = Counter((r['reviewer'], r['candidate'], int(r['round'])) for r in rows)
    if set(counts) != expected or any(v != 1 for v in counts.values()):
        raise ValueError('Expected exactly 72 unique ratings: four reviewers x six candidates x three rounds')
    for r in rows:
        if any(not 1 <= float(r[d]) <= 5 for d in DIMENSIONS) or not 6 <= float(r['source_total']) <= 30:
            raise ValueError('Rating outside the historical scale')
    scores, weights, _ = aggregate(rows)
    totals = []
    for candidate in sorted(candidates):
        subset = [r for r in rows if r['candidate'] == candidate]
        totals.append({'candidate': candidate, 'n_ratings': len(subset),
                       'mean_recorded_total': mean(float(r['source_total']) for r in subset),
                       'mean_dimension_sum': mean(sum(float(r[d]) for d in DIMENSIONS) for r in subset),
                       'total_discrepancy_count': sum(abs(sum(float(r[d]) for d in DIMENSIONS) - float(r['source_total'])) > 1e-8 for r in subset)})
    return scores, weights, sorted(totals, key=lambda r: -r['mean_recorded_total'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    scores, weights, totals = summarize(read_csv(args.ratings))
    if args.out_dir.exists():
        raise ValueError('Choose a new output directory to preserve previous summaries')
    args.out_dir.mkdir(parents=True)
    for filename, rows in [('iterative_scores.csv', scores), ('iterative_weights.csv', weights), ('mean_recorded_totals.csv', totals)]:
        write_csv(args.out_dir / filename, rows)
    print(f'Saved full-matrix aggregation to {args.out_dir}')


if __name__ == '__main__':
    main()
