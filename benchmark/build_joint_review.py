"""Build the historical joint-review format from one complete fresh system run."""
import argparse
import json
import re
from pathlib import Path

from run_benchmark import digest, load_jsonl

SOURCE_ORDER = ['Qwen3-4b', 'ChatGPT-4o', 'Deepseek-V3', 'RAMAD', 'ChatGPT-o3', 'RAMAD-RAG']
TABLE_ORDER = ['RAMAD', 'Deepseek-V3', 'RAMAD-RAG', 'ChatGPT-4o', 'Qwen3-4b', 'ChatGPT-o3']
ALIASES = {'Qwen3-4B': 'Qwen3-4b', 'DeepSeek-V3': 'Deepseek-V3', 'RAMAD-reconstructed': 'RAMAD'}


def build(rows, rubric):
    if len(rows) != 6 or len({r['candidate_label'] for r in rows}) != 6:
        raise ValueError('Expected exactly six unique candidate answers')
    if len({r['run_label'] for r in rows}) != 1 or len({r['question_id'] for r in rows}) != 1:
        raise ValueError('Answers must belong to one run and one question')
    if len({r['question'] for r in rows}) != 1:
        raise ValueError('Question text differs across answers')
    mapping = {ALIASES.get(r['candidate_label'], r['candidate_label']): r for r in rows}
    if set(mapping) != set(SOURCE_ORDER):
        raise ValueError('Six-model historical comparison requires all original candidate roles')
    for name, row in mapping.items():
        if row.get('record_type') != 'candidate_answer' or not row.get('answer', '').strip():
            raise ValueError(f'{name}: missing generated answer')
        if row.get('finish_reason') not in ('stop', 'eos_token'):
            raise ValueError(f'{name}: missing or incomplete generation finish status')
        expected = 'prompt_rag' if name in ('RAMAD', 'RAMAD-RAG') else 'bare'
        if row.get('condition') != expected:
            raise ValueError(f'{name}: inconsistent system-comparison condition')
        passages = row.get('retrieved_passages', [])
        if expected == 'prompt_rag' and len(passages) != 5:
            raise ValueError(f'{name}: expected five recovered passages')
        if expected == 'bare' and passages:
            raise ValueError(f'{name}: baseline unexpectedly received RAG passages')
        if row.get('retrieval_sha256') != digest(passages):
            raise ValueError(f'{name}: retrieval hash mismatch')
        if row.get('prompt_sha256') != digest(row.get('messages')):
            raise ValueError(f'{name}: prompt hash mismatch')
    if mapping['RAMAD']['retrieved_passages'] != mapping['RAMAD-RAG']['retrieved_passages']:
        raise ValueError('RAMAD and its RAG-only ablation must use identical passages')
    # Keep the historical six-dimensional request; change only candidate identities.
    display = {name: mapping[name]['candidate_label'] for name in TABLE_ORDER}
    pattern = '|'.join(re.escape(name) for name in sorted(display, key=len, reverse=True))
    request = re.sub(pattern, lambda m: display[m.group()], rubric).rstrip()
    request += '\n\nQuestion：' + rows[0]['question'] + '\n'
    for name in SOURCE_ORDER:
        row = mapping[name]
        request += '\n' + row['candidate_label'] + '：\n' + row['answer'].strip() + '\n'
        # Historical RAG answers included UI-appended retrieval evidence.
        if row['retrieved_passages']:
            request += '\n参考文献（检索系统附加内容）：\n'
            for i, passage in enumerate(row['retrieved_passages'], 1):
                source = passage.get('source_title') or passage.get('title') or passage.get('source') or passage['source_id']
                request += f"[{i}] {source}\n{passage['text'].strip()}\n"
    manifest = {'record_type': 'fresh_system_joint_review', 'run_label': rows[0]['run_label'],
                'question_id': rows[0]['question_id'], 'candidate_labels': [display[n] for n in TABLE_ORDER],
                'answer_order': [display[n] for n in SOURCE_ORDER],
                'answers_sha256': digest(rows), 'request_sha256': digest([{'role': 'user', 'content': request}]),
                'source_request_sha256': digest(rubric),
                'source_display': 'Five retrieval passages appended for the two RAG conditions, following the archived comparison.',
                'models': [{'label': r['candidate_label'], 'requested_model': r['candidate_model'],
                            'returned_model': r.get('returned_model'), 'prompt_sha256': r['prompt_sha256'],
                            'retrieval_sha256': r['retrieval_sha256']} for r in rows]}
    return request, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rubric = (root / 'historical/scoring_request_original_cn.txt').read_text(encoding='utf-8')
    request, manifest = build(load_jsonl(args.answers), rubric)
    if args.out_dir.exists():
        raise ValueError('Output directory exists; choose a new directory to preserve the previous request')
    args.out_dir.mkdir(parents=True)
    (args.out_dir / 'request.txt').write_text(request, encoding='utf-8')
    (args.out_dir / 'request_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved one six-candidate request to {args.out_dir}')


if __name__ == '__main__':
    main()
