# Completed six-model system reconstruction

This directory contains six generated answers, 12 joint-review responses and
72 ratings. RAMAD and RAMAD-RAG receive the recovered domain context and prompt;
the four general-model baselines receive the bare original question.

The reconstructed RAMAD ranks first under both the original iterative algorithm
(27.2381/30) and mean recorded totals (25.3333/30). The historical mean was 26.25.
All raw responses and arithmetic discrepancies are retained.

From the repository root:

```bash
python benchmark/summarize_joint_review.py --ratings benchmark/results/system/reviewer_rounds.csv --out-dir results/system_scores_recomputed
```

The configurations for this run are
`benchmark/benchmark_config.reconstructed_chat.reported.json` and
`benchmark/review_config.reported.json`. The full data/model companion package
contains the reconstructed LoRA adapter, base-model provenance and spectral
checkpoints. Historical records and the supplementary shared-context experiment
are stored separately.
