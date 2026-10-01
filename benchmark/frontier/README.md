# Frontier-model comparison code

This module reproduces the supplementary seven-model run reported in the
Supporting Information. The candidate panel contains GPT-5.5,
Gemini-3.1-Pro-Preview, ChatGPT-o3, Qwen3.5-27B, DeepSeek-V3, Grok-4 and
ChatGPT-4o. It does not contain a RAMAD response.

The five questions and model configuration used for the recorded run are in
this directory. Raw answers, evaluator responses and score tables are deposited
under `Zenodo模型与数据/benchmark/frontier/`.

Run a new comparison from this directory after setting `QINGYUN_API_KEY`:

```powershell
python run_frontier_eval.py generate --config run_config.json
python run_frontier_eval.py score --config run_config.json
python aggregate_scores.py --config run_config.json
```

The API endpoint in the configuration is an OpenAI-compatible third-party
gateway. Requested and returned model identifiers are retained in the raw
response records. Raw same-run totals must be reported separately from any
cross-run anchor transformation involving the historical RAMAD score.
