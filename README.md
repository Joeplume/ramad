# RAMAD

RAMAD combines domain-specific retrieval, prompting and a fine-tuned language
model for Raman/SERS experimental design, NLMixup spectral augmentation, and
CGANet classification and concentration prediction.

## Contents

| Directory | Contents |
| --- | --- |
| `benchmark/` | Questions, retrieved passages, prompts, model settings, generation and scoring code |
| `benchmark/frontier/` | Frontier-model comparison code (data records are in the companion package) |
| Companion `benchmark/framework_transfer/` | Kimi K3/GPT-5.5 bare-versus-RAMAD-framework records and joint scoring outputs |
| Companion `benchmark/equal_harness/` | RAMAD/Kimi K3/GPT-5.5 common-framework comparison |
| `benchmark/historical/` | Original experimental answers and scoring records |
| `benchmark/results/system/` | Six-model reproduction: six answers, 12 reviewer calls, 72 ratings and summaries |
| `src/ramad_model/` | LoRA training and inference |
| `src/ramad_rag/` | Document indexing, retrieval and question answering |
| `src/spectral/` | CGANet architecture and checkpoint evaluation |
| `../ramad_zenodo_package/training/` | Training corpus, splits and provenance |
| `training_examples/` | Additional training examples |
| `prompts/` | Generation and scoring templates |
| `interactive_scoring.py` | Manual entry and summarization of human expert scores |

Weights and spectral datasets are in the companion `ramad_zenodo_package`
directory. Download it from Zenodo (DOI: 10.5281/zenodo.23096172) and extract it
beside this repository. The adapter package is retained as a candidate artifact
pending author validation and is not presented as the historical Figure 2c
adapter.
For another installation layout, edit `adapter_path` in the benchmark configuration.

## Installation

Use Python 3.10. Install the spectral environment with:

```powershell
python -m pip install -r requirements-spectral.txt
```

For language-model training and inference, use a separate CUDA environment.
The recorded environment used PyTorch 2.9.1 with CUDA 12.8:

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-llm.txt
```

`environment-llm-windows-cu128.txt` records the installed package versions.
API generation and model review read `QINGYUN_API_KEY` from the environment.
Endpoint and requested model IDs are in the configuration; returned IDs are
recorded in the response logs. Offline score calculation needs no API key.

## Recompute recorded scores

Run from this repository directory:

```powershell
python benchmark/reproduce_historical_scores.py
python benchmark/summarize_joint_review.py --ratings benchmark/results/system/reviewer_rounds.csv --out-dir results/system_scores
```

The archived Figure 2c records give a recorded-total mean of 26.25/30 for
RAMAD. `iterative_scores.csv`, `iterative_weights.csv` and
`mean_recorded_totals.csv` are separate recalculations from the released
ratings and are not pooled with the framework-transfer experiment.

## Generate and score a new system comparison

`benchmark_config.main.json` is the default executable configuration. RAMAD uses
its adapter, domain prompt and five retrieved passages; RAMAD-RAG uses the same
prompt and passages with the base model; general-model baselines receive the
original question. Local chat serialization matches the released adapter's
training format. Parameter values and the base-model revision are recorded in
the configuration and model documentation.

```powershell
python benchmark/run_benchmark.py dry-run
python benchmark/run_benchmark.py generate
python benchmark/build_joint_review.py --answers benchmark/outputs/main/model_answers.jsonl --out-dir benchmark/outputs/joint_request
python benchmark/replay_historical_review.py score --config review_config.reported.json --request-file outputs/joint_request/request.txt --request-manifest outputs/joint_request/request_manifest.json --out-dir outputs/joint_review
python benchmark/summarize_joint_review.py --ratings benchmark/outputs/joint_review/reviewer_rounds.csv --out-dir results/new_system_scores
```

This joint review follows the original named six-candidate procedure, with four
reviewer families and three calls per reviewer. Full responses and parsed scores
are saved. The framework-transfer experiment applies the same five frozen RAG
passages, structured prompt and task constraints to Kimi K3 and GPT-5.5, with
bare-model responses retained as within-model controls. In the common
ten-candidate matrix, Kimi K3 increased from 20.44 to 25.67/30 and GPT-5.5
increased from 20.89 to 26.33/30. In the separate three-candidate equal-harness
matrix, RAMAD, Kimi K3 and GPT-5.5 scored 22.89, 26.33 and 23.33/30,
respectively. Complete records are archived in the companion
`benchmark/framework_transfer/` and `benchmark/equal_harness/` directories.
Claude is not assigned a score. `benchmark/BENCHMARK_PROTOCOL.md` defines the
two comparison panels and scoring procedure.

## Evaluate CGANet

The companion `spectral/cganet/` directory contains the August 2025 checkpoint,
preprocessing parameters, split assignments and 11,614 internal test spectra.
The released code reproduces the Figure 5b confusion matrix, including six STZ
samples classified as Water. The same evaluation gives RMSE 0.0479882 and
R² 0.9986685. See its README for the dataset and architecture details.

```powershell
python src/spectral/run_checkpoint.py --package-dir ../ramad_zenodo_package/spectral/cganet --out-dir results/cganet
```

For a new leakage-controlled training run, supply an explicit split table with
`source_row` and `split` (`train`, `val`, or `test`) columns. The split is applied
before preprocessing, and `RobustScaler` is fitted on training rows only:

```powershell
python src/spectral/train_cganet.py --training-csv PATH/augmented_multi_substances_filtered_water_m_cleaned.csv --split-csv PATH/batch_level_split.csv --out-dir results/cganet_training
```

The released August 2025 checkpoint and its archived row-level split are
retained only for recalculating the associated internal-test outputs. They are
not used as evidence for the batch-separated validation described in the
revised manuscript. New training runs must use the explicit batch-level split
workflow above. The source training table and checkpoint metadata are recorded
in `spectral/cganet/model_config.json`.

## Checks and troubleshooting

`pwsh -File run_public_package.ps1` checks dataset formats and CLI entry points.
It does not train models or make API calls.

- Missing adapter: check the companion directory and configured `adapter_path`.
- CUDA error: use a CUDA-enabled PyTorch build compatible with your GPU.
- API model unavailable: check the configured endpoint and model ID.
- Incomplete output: inspect the saved failure log before starting another run.
- Existing result directory: choose a new directory to retain previous records.

## LoRA training configuration

The training and single-prompt inference entry points default to
`training_config_candidate_qlora.json`. This candidate configuration remains
separate from the archived Figure 2c score record until author validation is
complete. `training_config.json` preserves the alternative full-precision
configuration.
