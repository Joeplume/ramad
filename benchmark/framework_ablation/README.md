# Frontier-model framework ablation

This experiment measures the effect of applying the RAMAD retrieval and structured-prompt layer to Claude Fable 5 and Kimi K3. It is separate from both the historical Figure 2c system comparison and the seven-model equal-harness control.

## Design

- Five archived Chinese SERS experimental-design questions are used in both conditions.
- The bare condition receives only the question.
- The prompt-RAG condition receives the same question plus the active RAMAD prompt and the same five frozen retrieved passages used in the equal-harness experiment.
- Generation uses temperature 0, top-p 1, and a maximum allowance of 16,384 output tokens.
- All four candidate conditions are blindly evaluated on the original six dimensions in three independent rounds by the same three executable reviewers: ChatGPT-o3, DeepSeek-V3, and ChatGPT-4o.
- Scores are aggregated with the manuscript's iterative per-dimension reviewer-weighting procedure. Prompt-RAG minus bare effects are paired by question (n = 5); the package reports exhaustive question bootstrap intervals and exact two-sided sign-flip tests.

The earlier seven-model equal-harness benchmark used an additional local Qwen3-4B reviewer. That local base-model weight was not available in the current workspace, and the API route for `qwen3-4b` returned an access-denied error. Therefore, this dedicated before-versus-after ablation uses the same three available reviewers for every condition and recomputes all four candidate scores within that common panel. Its totals should not be substituted for the seven-model totals.

## Results

| Model | Bare / 30 | RAMAD prompt-RAG / 30 | Gain | Relative gain |
| --- | ---: | ---: | ---: | ---: |
| Claude Fable 5 | 28.844 | 29.067 | +0.222 | +0.8% |
| Kimi K3 | 27.289 | 29.178 | +1.889 | +6.9% |

For Claude Fable 5, the paired-question bootstrap interval was -0.311 to 0.800 and the exact sign-flip p value was 0.625. For Kimi K3, the interval was 0.244 to 4.467 and p was 0.125. The small number of questions limits inferential power; these results support a numerical framework benefit, with a larger observed effect for Kimi K3, but not a claim of statistical significance.

## Files

- `benchmark_config.json`: candidate, reviewer, generation, and scoring configuration.
- `model_answers_bare.jsonl`: ten newly generated bare-condition answers.
- `model_answers.jsonl`: the four-condition answer matrix after adding the archived prompt-RAG answers.
- `model_scores_bare_round*.jsonl`: three raw blind-scoring rounds for the bare answers.
- `model_scores_round*.jsonl`: complete four-condition reviewer matrices.
- `model_scores.jsonl`: three-round mean scores.
- `summary/weighted_final_scores.csv`: four-condition weighted totals.
- `summary/per_question_weighted_scores.csv`: paired question-level scores.
- `summary/framework_effects.csv`: before-versus-after effects and paired statistics.
- `generate_kimi_stream.py`: streaming transport used to prevent proxy disconnects while preserving the frozen generation settings.
- `prepare_combined.py` and `summarize_framework_effect.py`: reproducible merge and effect-summary scripts.

## Reproduction

From `benchmark/`, configure `QINGYUN_API_KEY`, then run the bare generation and three scoring rounds with `run_benchmark.py`. Kimi generation may use `framework_ablation/generate_kimi_stream.py`; only the HTTP transfer mode differs. After scoring, run:

```bash
python framework_ablation/prepare_combined.py --equal-harness-dir <equal_harness_dir> --out-dir framework_ablation
python combine_reviewer_rounds.py --rounds framework_ablation/model_scores_round1.jsonl framework_ablation/model_scores_round2.jsonl framework_ablation/model_scores_round3.jsonl --out framework_ablation/model_scores.jsonl
python aggregate_scores.py --config framework_ablation/benchmark_config.json --scores framework_ablation/model_scores.jsonl --answers framework_ablation/model_answers.jsonl --questions questions.jsonl --out-dir framework_ablation/summary --run-label frontier-framework-ablation-cn-20261001
python framework_ablation/summarize_framework_effect.py --summary-dir framework_ablation/summary
```
