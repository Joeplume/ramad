# RAMAD

RAMAD combines domain-specific retrieval, prompting and a fine-tuned language
model for Raman/SERS experimental design, NLMixup spectral augmentation, and
CGANet classification and concentration prediction.

## Contents

| Directory | Contents |
| --- | --- |
| `benchmark/` | Questions, retrieved passages, prompts, model settings, generation and scoring code |
| `benchmark/historical/` | Original experimental answers and scoring records |
| `benchmark/results/system/` | Six-model reproduction: six answers, 12 reviewer calls, 72 ratings and summaries |
| `src/ramad_model/` | LoRA training and inference |
| `src/ramad_rag/` | Document indexing, retrieval and question answering |
| `src/spectral/` | CGANet preprocessing, training and checkpoint inference |
| `../Zenodo模型与数据/training/` | Training corpus, splits and provenance |
| `training_examples/` | Additional training examples |
| `prompts/` | Generation and scoring templates |

Weights and spectral datasets are in the companion `Zenodo模型与数据` directory.
Keep that directory beside this repository to use the supplied adapter paths.
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

The original recorded-total mean is 26.25/30 for RAMAD. The September 2026
reproduction gives 25.3333/30 by the same recorded-total mean and 27.2381/30
by the original iterative weighting. Its complete six-model table is in the
results directory. `iterative_scores.csv`, `iterative_weights.csv` and
`mean_recorded_totals.csv` are separate calculations from the supplied ratings.

## Generate and score a new system comparison

`benchmark_config.main.json` is the default executable configuration. RAMAD uses
its adapter, domain prompt and five retrieved passages; RAMAD-RAG uses the same
prompt and passages with the base model; general-model baselines receive the
original question. Local chat serialization matches the released adapter's
training format. Parameter values, base revision and reconstruction settings
are recorded in the configuration and model documentation.

```powershell
python benchmark/run_benchmark.py dry-run
python benchmark/run_benchmark.py generate
python benchmark/build_joint_review.py --answers benchmark/outputs/main_chat_reconstruction/model_answers.jsonl --out-dir benchmark/outputs/joint_request
python benchmark/replay_historical_review.py score --config review_config.reported.json --request-file outputs/joint_request/request.txt --request-manifest outputs/joint_request/request_manifest.json --out-dir outputs/joint_review
python benchmark/summarize_joint_review.py --ratings benchmark/outputs/joint_review/reviewer_rounds.csv --out-dir results/new_system_scores
```

This joint review follows the original named six-candidate procedure, with four
reviewer families and three calls per reviewer. Full responses and parsed scores
are saved. `benchmark/frontier/` contains the executable code for the seven-model
frontier run reported in the Supporting Information. The equal-harness control
supplies identical prompts and retrieved passages to seven candidates on five
questions. Its complete records and paired statistics are in
`../Zenodo模型与数据/benchmark/equal_harness/README.md`.

## CGANet

The companion dataset contains the 1,600-channel mixed-drug model and the
1,800-channel four-class model, each with its checkpoint, preprocessing and split.
Run the mixed-drug checkpoint using:

```powershell
python src/spectral/run_checkpoint.py --package-dir ../Zenodo模型与数据/spectral/cganet_1600 --out-dir results/spectral/cganet_1600
```

Training and preprocessing commands are included in the companion README.
For the four-class model, see `spectral/cganet_1800/README.md` there.
NLMixup-generated spectra are training augmentation data; external prediction
files identify the measured test inputs and their labels.

## Checks and troubleshooting

`pwsh -File run_public_package.ps1` checks dataset formats and CLI entry points.
It does not train models or make API calls. The two-spectrum CSV is a format
sample; full checkpoint evaluation uses the companion spectral datasets.

- Missing adapter: check the companion directory and configured `adapter_path`.
- CUDA error: use a CUDA-enabled PyTorch build compatible with your GPU.
- API model unavailable: check the configured endpoint and model ID.
- Incomplete output: inspect the saved failure log before starting another run.
- Existing result directory: choose a new directory to retain previous records.

## LoRA training configuration

The training and single-prompt inference entry points default to
`training_config_reconstruction_qlora.json`, which matches the released adapter's
recorded training settings. `training_config.json` preserves the earlier training
configuration. The model card and `run_manifest.json` identify the completed run.
