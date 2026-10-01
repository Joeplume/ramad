import argparse
import csv
import json
import re
from pathlib import Path

from run_benchmark import (
    append_jsonl, completion, digest, enabled_models, finish_reason,
    load_json, load_jsonl, saved_parameters, utc_now, visible_text,
)

DIMENSIONS = ['SR', 'CSC', 'DQ', 'CS', 'QR', 'IS']
CANDIDATES = ['RAMAD', 'Deepseek-V3', 'RAMAD-RAG', 'ChatGPT-4o', 'Qwen3-4b', 'ChatGPT-o3']


def canonical_label(label, candidates=CANDIDATES):
    clean = re.sub(r'[^a-z0-9]', '', label.lower())
    lookup = {re.sub(r'[^a-z0-9]', '', name.lower()): name for name in candidates}
    return lookup.get(clean)


def parse_table(text, candidates=CANDIDATES):
    result = {}
    for line in text.splitlines():
        if '|' not in line:
            continue
        cells = [c.strip().replace('**', '').replace('`', '') for c in line.strip().strip('|').split('|')]
        if len(cells) != 8:
            continue
        label = canonical_label(cells[0], candidates)
        if label is None:
            continue
        if label in result:
            raise ValueError(f'Duplicate candidate row: {label}')
        if not all(re.fullmatch(r'\d+(?:\.\d+)?', c) for c in cells[1:]):
            raise ValueError(f'Non-numeric rating: {label}')
        values = [float(c) for c in cells[1:]]
        if not all(1 <= value <= 5 for value in values[:6]) or not 6 <= values[6] <= 30:
            raise ValueError(f'Rating outside source scale: {label}')
        result[label] = {'candidate': label, **dict(zip(DIMENSIONS, values[:6])),
                         'source_total': values[6], 'dimension_sum': sum(values[:6]),
                         'total_difference': sum(values[:6])-values[6]}
    if set(result) != set(candidates):
        raise ValueError(f'Incomplete candidate table: {sorted(set(candidates)-set(result))}')
    return [result[name] for name in candidates]


def main():
    parser = argparse.ArgumentParser(description='Replay the preserved six-candidate joint review request.')
    parser.add_argument('command', choices=['dry-run', 'score', 'export'])
    parser.add_argument('--config', default='historical_review.config.example.json')
    parser.add_argument('--request-file', default='historical/archived_comparison_text.txt')
    parser.add_argument('--request-manifest', help='Manifest from build_joint_review.py for fresh generated answers')
    parser.add_argument('--out-dir', default='outputs/historical_review_replay')
    parser.add_argument('--evaluator-label', action='append')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_json(root / args.config)
    request = (root / args.request_file).read_text(encoding='utf-8')
    evaluators = enabled_models(config, 'evaluator_models', args.evaluator_label)
    if not evaluators:
        raise ValueError('No reviewers enabled')
    rounds = int(config.get('review_rounds', 3))
    if rounds != 3:
        raise ValueError('Historical protocol requires three rounds')
    run_label = config['run_label']
    messages = [{'role': 'user', 'content': request}]
    request_hash = digest(messages)
    candidates = CANDIDATES
    if args.request_manifest:
        manifest = load_json(root / args.request_manifest)
        if manifest['request_sha256'] != request_hash:
            raise ValueError('Request differs from its manifest')
        candidates = manifest['candidate_labels']
        if len(candidates) != 6 or len(set(candidates)) != 6:
            raise ValueError('Expected six unique candidates in manifest')
    out = root / args.out_dir
    log = out / 'raw_joint_reviews.jsonl'
    if args.command == 'dry-run':
        print(json.dumps({'run_label': run_label, 'request_sha256': request_hash,
                          'request_characters': len(request), 'scoring_table_order': candidates,
                          'reviewers': [r['label'] for r in evaluators], 'rounds': rounds,
                          'planned_calls': len(evaluators)*rounds, 'output': str(out)}, indent=2))
        return
    previous = load_jsonl(log) if log.exists() else []
    if any(r['request_sha256'] != request_hash or r['run_label'] != run_label for r in previous):
        raise ValueError('Output directory contains a different request or run; select a new directory')
    done = {(r['reviewer'], r['round']): r for r in previous}
    if len(done) != len(previous):
        raise ValueError('Duplicate stored reviewer-round calls')
    if args.command == 'score':
        for evaluator in evaluators:
            settings_hash = digest({'reviewer': evaluator, 'scoring': config.get('scoring', {})})
            for round_id in range(1, rounds+1):
                key = (evaluator['label'], round_id)
                if key in done:
                    old = done[key]
                    if old['settings_sha256'] != settings_hash or old['status'] != 'complete':
                        raise ValueError('Existing call is failed or has different settings; inspect it before a new run')
                    continue
                response, profile, parameters = completion(config, evaluator, messages, 'scoring')
                text = visible_text(response)
                status, error, parsed = 'complete', '', []
                try:
                    if not text.strip() or finish_reason(response) == 'length':
                        raise ValueError('Empty or truncated review')
                    parsed = parse_table(text, candidates)
                except ValueError as exc:
                    status, error = 'invalid', str(exc)
                row = {'run_label': run_label, 'timestamp_utc': utc_now(),
                       'reviewer': evaluator['label'], 'round': round_id,
                       'request_sha256': request_hash, 'settings_sha256': settings_hash,
                       'messages': messages, 'requested_model': evaluator['model'],
                       'api_profile': profile, 'parameters': saved_parameters(parameters),
                       'response': response, 'status': status, 'error': error, 'parsed_scores': parsed}
                append_jsonl(log, row)
                if error:
                    raise ValueError(f'{key}: {error}; complete raw response saved')
                print(f"Saved {evaluator['label']} round {round_id}", flush=True)
    records = load_jsonl(log)
    expected = {(r['label'], n) for r in evaluators for n in range(1, rounds+1)}
    selected = [r for r in records if r['reviewer'] in {e['label'] for e in evaluators}]
    if {(r['reviewer'], r['round']) for r in selected} != expected or any(r['status'] != 'complete' for r in selected):
        raise ValueError('Reviewer matrix is incomplete or contains invalid calls')
    rows = [{ 'reviewer': r['reviewer'], 'round': r['round'], **score}
            for r in selected for score in r['parsed_scores']]
    with (out / 'reviewer_rounds.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Exported {len(rows)} fresh ratings. Archived ratings were not changed.')


if __name__ == '__main__':
    main()
