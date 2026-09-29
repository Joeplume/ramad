# Shared-context RAMAD benchmark protocol

## Comparison question

Does the released, domain-tuned RAMAD adapter improve Raman/SERS experimental-design answers when all models receive the same domain instruction, the same retrieved literature passages, and the same five questions? The comparison uses one frozen retrieval result for each question. The prompts in `../prompts/` are the exact templates for this new comparison run; they are not represented as a verbatim record of an earlier historical run.

## Conditions

The primary comparison is RAMAD-LoRA versus Claude Fable 5 and Kimi K3, each with the same system instruction and the same five retrieved passages in the same order. The local Qwen3-4B base model is also evaluated with that shared prompt and context to isolate the effect of the LoRA adapter. The additional local base-model arms without a domain prompt and with the prompt but no retrieval show how much those framework components contribute within the local backbone. Model IDs, revisions, provider, generation settings, request timestamps, returned model IDs, messages, passages, and responses are saved for the run.

Use the five questions in `questions.jsonl`. Build the index with the parameters in `retrieval_config.json`, then run `freeze_retrieval.py` once. Inspect and archive the resulting `retrieval_contexts.jsonl` before generating any answer. Every model in a `prompt_rag` arm receives those identical passages; no model-specific search or hand-picked evidence is allowed. The common output-token cap is 4,096; this includes hidden reasoning tokens where a provider uses them. An empty or truncated visible answer is logged as a failed call. If that occurs, choose a larger common cap, assign a new run label, and regenerate the complete candidate matrix before scoring. Record any unavailable model before the run, without substituting one after viewing answers.

## Scoring

Three Raman/SERS or analytical-chemistry experts independently score responses under blinded IDs. They see the question, the fixed retrieved passages, and the answer, but not the model label. Each dimension is scored from 1 to 5, with 1 poor, 3 adequate, and 5 excellent:

| Code | Dimension | Scoring focus |
| --- | --- | --- |
| SR | Scenario relevance | Does the answer address fishpond water, target residues, field constraints, and the specific task? |
| CSC | Citation support and credibility | Are factual and literature claims supported by the supplied passages, and are uncertainty and limits stated without invented data? |
| DQ | Design quality | Is the proposed analytical workflow technically feasible and sufficiently specified? |
| CS | Clarity and structure | Can a reader follow the steps and their rationale? |
| QR | Question responsiveness | Does the answer cover the requested components directly and completely? |
| IS | Insightfulness | Does it offer useful, justified optimization ideas beyond generic statements? |

Experts use the same rubric and record their six numeric scores in `expert_scores.csv` from the generated template. A brief calibration on answers outside the test set can align interpretation of the scale, but the test responses are scored independently before labels are revealed. The primary score is the mean of the three expert ratings in each dimension, summed to a 30-point total and averaged over questions. LLM-judge ratings, if collected, are secondary and are not used to replace missing expert scores.

The paired comparison uses each question as the unit: RAMAD's total minus the comparator's total on that question. Report the mean paired difference, question bootstrap 95% interval, and a two-sided exact sign-flip p-value. With five questions, this analysis is exploratory; report the interval and the individual question scores alongside any p-value. Do not merge scores from an older round with this run. Do not report a model ranking until the full answer matrix and all three experts' score rows are present.

## Release record

Archive `benchmark_config.json`, `questions.jsonl`, `retrieval_config.json`, `retrieval_contexts.jsonl`, the two prompt files, `outputs/model_answers.jsonl`, blinded response sheet, completed expert scoring file, analysis outputs, code revision, and the corresponding adapter artifact together. Credentials are supplied through environment variables and are never archived.
