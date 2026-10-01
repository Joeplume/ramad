import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

DIMENSIONS = ['SR', 'CSC', 'DQ', 'CS', 'QR', 'IS']
REVIEWERS = ['ChatGPT-o3', 'Deepseek-V3', 'Qwen3-4b', 'ChatGPT-4o']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    aliases = {'DeepSeek-V3': 'Deepseek-V3', 'Qwen3-4B': 'Qwen3-4b'}
    rows = [{**row, 'reviewer': aliases.get(row['reviewer'], row['reviewer']),
             'candidate': aliases.get(row['candidate'], row['candidate'])} for row in rows]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['reviewer'], row['candidate']].append(row)
    candidates = sorted({row['candidate'] for row in rows})
    matrix = {}
    for reviewer in REVIEWERS:
        for candidate in candidates:
            values = grouped[reviewer, candidate]
            if not values:
                raise ValueError(f'Missing reviewer-candidate pair: {reviewer}, {candidate}')
            matrix[reviewer, candidate] = {d: mean(float(r[d]) for r in values) for d in DIMENSIONS}
    result = {c: {'candidate': c} for c in candidates}
    weights = []
    for dim in DIMENSIONS:
        alpha = [0.25] * 4
        for iteration in range(1, 201):
            scores = {c: sum(alpha[i] * matrix[r, c][dim] for i, r in enumerate(REVIEWERS)) for c in candidates}
            reviewer_scores = [scores[r] for r in REVIEWERS]
            lo, hi = min(reviewer_scores), max(reviewer_scores)
            z = [(v-lo)/(hi-lo) + 1e-6 for v in reviewer_scores] if hi != lo else [0.25 + 1e-6] * 4
            total = sum(z)
            new_alpha = [0.95 * v / total + 0.05 / 4 for v in z]
            delta = sum(abs(a-b) for a, b in zip(alpha, new_alpha))
            alpha = new_alpha
            if delta < 1e-7:
                break
        for c in candidates:
            result[c][dim] = sum(alpha[i] * matrix[r, c][dim] for i, r in enumerate(REVIEWERS))
        weights.append({'dimension': dim, 'iterations': iteration, **dict(zip(REVIEWERS, alpha))})
    for row in result.values():
        row['total_30'] = sum(row[d] for d in DIMENSIONS)
    return sorted(result.values(), key=lambda r: -r['total_30']), weights, matrix


def main():
    parser = argparse.ArgumentParser(description='Recompute preserved historical scores without model calls.')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = args.out_dir or root / 'historical' / 'recomputed'
    out.mkdir(parents=True, exist_ok=True)
    sources = {
        'source_script_rounded_means': root / 'historical_reviewer_scores.csv',
        'source_doc_exact_round_means': root / 'historical' / 'reviewer_rounds.csv',
    }
    matrices = {}
    for name, path in sources.items():
        scores, weights, matrix = aggregate(read_csv(path))
        write_csv(out / f'{name}_scores.csv', scores)
        write_csv(out / f'{name}_weights.csv', weights)
        matrices[name] = matrix
        print(name)
        for row in scores:
            print(f"  {row['candidate']}: {row['total_30']:.6f}")
    discrepancies = []
    first, second = matrices.values()
    for key in first:
        for dim in DIMENSIONS:
            delta = second[key][dim] - first[key][dim]
            if abs(delta) > 0.005:
                discrepancies.append({'reviewer': key[0], 'candidate': key[1], 'dimension': dim,
                                      'script_value': first[key][dim], 'round_mean': second[key][dim], 'difference': delta})
    if discrepancies:
        write_csv(out / 'source_matrix_differences.csv', discrepancies)
    print(f'Source differences beyond rounding: {len(discrepancies)}')
    rounds = read_csv(sources['source_doc_exact_round_means'])
    paper_table = read_csv(root / 'historical' / 'manuscript_table_source.csv')
    reconciliation = []
    for paper_row in paper_table:
        candidate = paper_row['candidate']
        ratings = [r for r in rounds if r['candidate'] == candidate]
        if len(ratings) != 12:
            raise ValueError(f'Expected 12 historical ratings for {candidate}')
        recorded_total = mean(float(r['source_total']) for r in ratings)
        rec = {'candidate': candidate, 'n_ratings': len(ratings),
               'mean_recorded_total': recorded_total,
               'paper_table_total': float(paper_row['total_30']),
               'rounded_total_matches': round(recorded_total, 2) == float(paper_row['total_30']),
               'mean_dimension_sum': mean(float(r['dimension_sum']) for r in ratings)}
        for dim in DIMENSIONS:
            rec[f'{dim}_round_mean'] = mean(float(r[dim]) for r in ratings)
            rec[f'{dim}_table_value'] = float(paper_row[dim])
        reconciliation.append(rec)
    write_csv(out / 'manuscript_total_reconciliation.csv', reconciliation)
    if not all(r['rounded_total_matches'] for r in reconciliation):
        raise ValueError('A manuscript table total does not match archived recorded totals')
    print('All six manuscript table totals reproduced from mean recorded totals.')


if __name__ == '__main__':
    main()
