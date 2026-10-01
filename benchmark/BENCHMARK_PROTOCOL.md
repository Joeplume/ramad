# Supplementary shared-RAG benchmark protocol

This protocol applies only to the supplementary shared-context experiment. The main system comparison is configured in `benchmark_config.main.json`. The recovered historical question, six answers, original joint evaluation request, and ratings are documented in `historical/README.md`.

## Source materials

The five Chinese questions in `questions.jsonl` are taken from the archived `细分问题打分.docx`. Every candidate receives the active Chinese RAG prompt recovered from the original `rag_qa.py`, with the same five passage texts inserted in the same order. No additional domain system prompt or passage-number instruction is supplied to any candidate.

The passages were frozen from the preserved FAISS index before generation. `retrieval_queries_en.jsonl` records the English retrieval queries corresponding to the five Chinese questions; the preserved index was built with `all-MiniLM-L6-v2`, so these queries are used consistently for every candidate. `retrieval_config.json` records the encoder, top-k value, and index hashes. `retrieval_contexts.jsonl` stores the exact passages and source-file identifiers; only passage text enters the candidate prompt, matching the original RAG template.

## Generation

The candidate panel contains the newly reconstructed RAMAD LoRA adapter, its untuned Qwen3-4B base, Claude Fable 5, Kimi K3, ChatGPT o3, DeepSeek V3, and ChatGPT 4o. All seven receive the same questions, prompt template, retrieved passages, temperature 0, and maximum output-token setting of 16,384. Calls that return empty or truncated visible answers are excluded and repeated under a new complete-run label. The provider, actual returned model ID, request parameters, messages, response, and usage are saved for every call.

The reconstructed adapter is a separately labeled model trained from the supplied reconstructed 1,000-record corpus. It is not the historical adapter behind the manuscript's original Figure 2c. Historical scores and this new run are not pooled.

## Scoring and analysis

The four reviewer model families named in the manuscript—ChatGPT o3, DeepSeek V3, Qwen3-4B, and ChatGPT 4o—score blinded answers on the original six dimensions: SR, CSC, DQ, CS, QR, and IS, each on a 1–5 scale. `prompts/scoring_rubric_cn.txt` states the operational rubric. Each reviewer-answer pair is scored in three recorded calls. The three scores are averaged separately for each dimension before the manuscript's iterative per-dimension reviewer weighting is applied. Raw calls, the mean score matrix, reviewer weights, and arithmetic-mean sensitivity scores are retained.

The Qwen3-4B reviewer uses its non-thinking chat-template mode for concise six-dimension JSON output in all three rounds. Candidate-model Qwen3-4B and RAMAD answers retain their default inference mode. A preflight reviewer run with thinking enabled was excluded in full after one answer exhausted the 8,192-token scoring cap without producing a final rating.

Human expert scores, if collected, are recorded separately and combined with the LLM average according to Text S7. They are never inferred from LLM scores or copied from the historical model table. The paired statistical unit is one question (n = 5); report per-question scores, mean paired differences, question bootstrap intervals, and exact two-sided sign-flip tests without treating the five questions as a large sample.

## Reproduction record

The release contains the Chinese questions, English retrieval queries, frozen passages, exact generation and scoring prompts, model settings, all answer and scoring logs, the aggregation code, and the reconstructed adapter identifier. The full-text PDF archive is not required to recompute the published scores because the exact supplied passages are frozen and released.
