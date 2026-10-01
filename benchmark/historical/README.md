# Historical scoring records

`reviewer_rounds.csv` contains 72 ratings recovered from the archived scoring document: four reviewer models, six candidates, and three rounds. Each reviewer scored all six candidates, including itself. The companion comparison document contains one overall-design question and six named answers. The five separately archived question variants are not substituted for that original question.

`archived_answers.jsonl` preserves the six answers with appended source excerpts. `question_original.jsonl` contains the original question. `scoring_request_original_cn.txt` preserves the full original joint-scoring request, including model labels and its instruction to differentiate scores. This historical procedure is not labeled as blind scoring. `archived_comparison_text.txt` preserves the full extracted text for checking the transcription.

The five source excerpts appended to RAMAD and RAMAD-RAG are identical. Each matched a passage in the preserved index after removing newlines exactly as the original web interface did. `../main_retrieval_contexts.jsonl` restores the indexed passage texts and their order, with matching document IDs and text hashes. No translated query or new relevance ranking was used to recover these passages.

To inspect a replay of the original joint-scoring request:

```bash
python benchmark/replay_historical_review.py dry-run
```

After configuring reviewer endpoints and credentials, `score` performs twelve calls (four reviewer families, three rounds). Full requests, parameters, returned model IDs, responses, and parsed scores are retained. Duplicate, missing, out-of-scale, empty, or truncated ratings stop export. New reviews are stored in `outputs/historical_review_replay/`; they do not replace historical ratings. Current service versions and explicit local reviewer reconstruction settings may differ from those of the historical run.

`scoring_rubric_original_cn.txt` preserves the six original criteria. `source_summary_tables.csv` preserves the document's LLM summary, human aggregate, and combined summary as separate tables. Individual expert ratings are not included in the source table. Source total columns are retained verbatim; independently calculated dimension sums are supplied alongside them.

Run from the repository root:

```bash
python benchmark/reproduce_historical_scores.py
```

The script reproduces the archived per-dimension iterative weighting using maximum 200 iterations, tolerance 1e-7, epsilon 1e-6, and damping 0.05. It computes both the original script's rounded reviewer means and the recovered three-round means without rounding. No model calls are needed.

| Candidate | Archived script means | Recovered round means |
| --- | ---: | ---: |
| RAMAD | 26.100138 | 26.099491 |
| Deepseek-V3 | 25.858780 | 25.856988 |
| ChatGPT-o3 | 23.091123 | 23.094257 |
| ChatGPT-4o | 22.018410 | 22.020230 |
| RAMAD-RAG | 20.265015 | 20.262374 |
| Qwen3-4b | 18.144804 | 18.150702 |

The manuscript table is preserved in `manuscript_table_source.csv`, transcribed from `多模型问题回答对照最终结果.xlsx`, Sheet1!K15:R20. Its six total scores are the arithmetic means of the 12 recorded total-score entries per candidate, rounded to two decimal places: RAMAD 26.25, Deepseek-V3 25.33, ChatGPT-o3 23.08, ChatGPT-4o 21.67, RAMAD-RAG 19.58, and Qwen3-4b 19.42. `recomputed/manuscript_total_reconciliation.csv` checks this mapping.

The table totals and the per-dimension iterative scores above are distinct archived calculations. Some source total entries do not equal the sum of their six dimensions; both are preserved. The reconciliation exports dimension means alongside the table's dimension values. No source cells or ratings are overwritten to reconcile these differences.

These are archived records, not fresh model inference results. Human aggregate rows have not been reassigned to new responses or substituted for individual expert scores.
