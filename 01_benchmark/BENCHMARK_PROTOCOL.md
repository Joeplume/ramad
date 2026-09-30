# Shared-context RAMAD benchmark protocol

## Comparison question

Does the released, domain-tuned RAMAD adapter improve Raman/SERS experimental-design answers when all models receive the same domain instruction, the same retrieved literature passages, and the same five questions? The comparison uses one frozen retrieval result for each question. `../prompts/domain_system.txt` and `../prompts/rag_user.txt` are the exact templates for this new comparison run; they are not represented as a verbatim record of an earlier historical run. `../prompts/original_rag_qa_template_2025.txt` preserves the recovered active template from the archived RAG script for provenance.

## Conditions

The primary comparison is RAMAD-LoRA versus Claude Fable 5 and Kimi K3, each with the same system instruction and the same five retrieved passages in the same order. The local Qwen3-4B base model is also evaluated with that shared prompt and context to isolate the effect of the LoRA adapter. The original four reviewer model families are included as common-prompt anchor candidates for the paper's iterative weighting calculation. Optional local base-model arms without a domain prompt and with the prompt but no retrieval can show how much those framework components contribute within the local backbone. Qwen3 local generation uses `enable_thinking=false` so the saved answer is the final response. Model IDs, revisions, provider, generation settings, request timestamps, returned model IDs, messages, passages, and responses are saved for the run.

Use the five questions in `questions.jsonl`. Query the recovered original index with the verified encoder in `retrieval_config.json`, then run `freeze_retrieval.py` once. Inspect and archive the resulting `retrieval_contexts.jsonl` before generating any answer. Every model in a `prompt_rag` arm receives those identical passages; no model-specific search or hand-picked evidence is allowed. The final common output-token cap is 16,384; this includes hidden reasoning tokens where a provider uses them. Earlier 4,096- and 8,192-token pilot calls were marked truncated and excluded. An empty or truncated visible answer is logged as a failed call. If that occurs, choose a larger common cap, assign a new run label, and regenerate the complete candidate matrix before scoring. Record any unavailable model before the run, without substituting one after viewing answers.

## Scoring

As in the paper's original evaluation, multiple LLM reviewers score responses under blinded IDs. They see the question, the fixed retrieved passages, and the answer, but not the model label. The six dimensions retain the paper's names and 1-to-5 scale, where 1 is poor, 3 is adequate, and 5 is excellent:

| Code | Dimension | Scoring focus |
| --- | --- | --- |
| SR | Scenario relevance | Does the answer address fishpond water, target residues, field constraints, and the specific task? |
| CSC | Citation support and credibility | Are factual and literature claims supported by the supplied passages, and are uncertainty and limits stated without invented data? |
| DQ | Design quality | Is the proposed analytical workflow technically feasible and sufficiently specified? |
| CS | Clarity and structure | Can a reader follow the steps and their rationale? |
| QR | Question responsiveness | Does the answer cover the requested components directly and completely? |
| IS | Insightfulness | Does it offer useful, justified optimization ideas beyond generic statements? |

The four archived reviewer model families (ChatGPT-o3, DeepSeek-V3, Qwen3-4B, and ChatGPT-4o) are run with the same scoring prompt and generation cap. Save every judge response and all six raw scores. Report both the arithmetic mean across judges and the per-dimension iterative weighting used for the historical analysis, including the new weights and convergence logs. These are new-run scores and must not be numerically merged with the historical RAMAD table. An optional independent human blind review can check answer quality; it is labeled separately from the paper's LLM-reviewer protocol.

The paired comparison uses each question as the unit: RAMAD's total minus the comparator's total on that question. Report the mean paired difference, question bootstrap 95% interval, and a two-sided exact sign-flip p-value. With five questions, this analysis is exploratory; report the interval and the individual question scores alongside any p-value. Do not merge scores from an older round with this run. Do not report a model ranking until the full answer matrix and all four reviewer score rows are present.

## Release record

Archive `benchmark_config.json`, `questions.jsonl`, `retrieval_config.json`, `retrieval_contexts.jsonl`, the two comparison prompt files, `outputs/model_answers.jsonl`, the four reviewer response logs, raw score matrix, weighted and unweighted analysis outputs, code revision, and the corresponding adapter artifact together. Credentials are supplied through environment variables and are never archived.
