# RAMAD reproducibility package

RAMAD combines a domain-specific literature retrieval and prompting assistant for Raman/SERS experimental planning with NLMixup spectral augmentation and CGANet spectral prediction. The LLM benchmark evaluates written experimental-design recommendations. The released spectral checkpoint supports a separate prediction workflow.

## Repository contents

| Path | Purpose |
| --- | --- |
| `01_benchmark/` | Fixed questions, shared-RAG generation, frozen retrieval records, blind scoring, and paired analysis |
| `02_code_entrypoints/ramad_model/` | Qwen3-4B LoRA training and inference with a saved 950/50 record split |
| `02_code_entrypoints/ramad_rag/` | Local PDF indexing, retrieval, and question answering |
| `02_code_entrypoints/spectral/` | CGANet architecture, preprocessing, training, and checkpoint inference |
| `04_training_corpus_1000/` | The 1,000-record public instruction dataset |
| `prompts/` | The exact domain instruction and retrieved-evidence prompt used by the new comparison |

The 7,657-publication corpus described in the paper is not included because source PDFs may have access restrictions. An authorized copy of the corpus or its FAISS index and a DOI mapping are needed to recreate the paper-scale retrieval condition. The frozen passages used in a reported benchmark must be archived with that benchmark, subject to source permissions. The original RAMAD LoRA adapter is not present in this repository; training code alone does not reproduce the reported model. Model artifacts and spectral data are prepared separately for a versioned Zenodo deposit. The earlier Zenodo record, [10.5281/zenodo.17809450](https://doi.org/10.5281/zenodo.17809450), contains one file named `model-00001-of-00004.safetensors` and is not an executable adapter package.

## Installation

Use Python 3.10 or later. The spectral checkpoint was verified with the versions in `requirements-spectral.txt`. For LoRA training and local generation, create a separate GPU environment, install the CUDA-enabled PyTorch build appropriate for that GPU, then install `requirements-llm.txt`. For example, PyTorch provides a CUDA 12.8 wheel for Windows in its [versioned installation guide](https://pytorch.org/get-started/previous-versions/). Set `QINGYUN_API_KEY` only when using the API models in `01_benchmark/benchmark_config.example.json`; keep credentials out of configuration files and released logs.

```powershell
python -m pip install -r requirements-spectral.txt
pwsh -File run_public_package.ps1
```

For the checked Windows RTX 5060 environment, install the CUDA 12.8 build of PyTorch 2.9.1 in the separate GPU environment, then install the pinned LLM packages:

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-llm.txt
```

`environment-llm-windows-cu128.txt` records the 117-package environment used for the CUDA import check; `pip check` found no broken requirements.

The published model ID and exact Qwen base revision are recorded in `02_code_entrypoints/ramad_model/training_config.json`. The training input is the full 1,000-record corpus; the saved train and validation IDs are in `02_code_entrypoints/ramad_model/splits/`.

## Reproduce the LLM comparison

1. Build the literature index from the authorized PDF set with `02_code_entrypoints/ramad_rag/build_index.py`. Its embedding model, token-based chunking, and top-five retrieval settings are listed in `01_benchmark/retrieval_config.json`.
2. Fill `01_benchmark/doi_mapping.example.csv` with real source filenames, DOIs, and titles. Run `01_benchmark/freeze_retrieval.py` to write one fixed `retrieval_contexts.jsonl` record per question. Use the same record for every `prompt_rag` candidate.
3. Copy `01_benchmark/benchmark_config.example.json` to `01_benchmark/benchmark_config.json`. Replace the RAMAD adapter path with the released directory. Verify the exact API model IDs and save the access date and provider in the run record.
4. Run the commands below. The `dry-run` lists the complete model and question matrix before any model call. `generate` records the actual messages, retrieved passages, parameters, response, returned model ID, and usage for each call.
5. Give the blinded response sheet and six-dimension rubric to three domain experts. After their independent scores are entered, `human_primary.py analyze` creates the model table, question-paired differences, bootstrap intervals, and an exact sign-flip test. The optional LLM-as-judge script is secondary to expert scoring.

```powershell
python 01_benchmark/freeze_retrieval.py --index-dir data/faiss_index --doi-map data/doi_mapping.csv --allow-pickle
python 01_benchmark/run_benchmark.py dry-run --config benchmark_config.json
python 01_benchmark/run_benchmark.py generate --config benchmark_config.json
python 01_benchmark/human_primary.py prepare
python 01_benchmark/human_primary.py analyze --scores 01_benchmark/outputs/expert_scores.csv
```

The five existing questions are in `01_benchmark/questions.jsonl`. The common conditions, six scoring dimensions, and analysis plan are specified in `01_benchmark/BENCHMARK_PROTOCOL.md`. Use one frozen run label and one scoring round for all candidates. Do not combine historical RAMAD scores with newly generated frontier-model scores. The complete answer and expert-scoring matrix is required before reporting a model ranking.

## Reproduce the CGANet checkpoint run

Download the `cganet_mixed_1600` files from the versioned Zenodo release into a local directory. Run the commands below to rebuild its scaler and row split and to evaluate the 51 external spectra. `run_checkpoint.py` writes per-sample predictions and a metric summary.

```powershell
python 02_code_entrypoints/spectral/prepare_mixed_1600.py --data-dir data/cganet_mixed_1600 --out-dir data/cganet_mixed_1600
python 02_code_entrypoints/spectral/run_checkpoint.py --package-dir data/cganet_mixed_1600 --out-dir results/cganet_mixed_1600
python 02_code_entrypoints/spectral/train_mixed_1600.py --package-dir data/cganet_mixed_1600 --out-dir results/cganet_retraining --epochs 200
```

The checkpoint corresponds to a 1,600-channel mixed-drug model with category labels MG, OFX, and STZ. Its saved 80/20 internal partition uses seed 42 and selects a checkpoint by internal loss; the external set is `test_data_250218.csv`. For the checked source files, external inference returns 48 correct classifications among 51 spectra and concentration RMSE 0.3712 in the source concentration scale. Retraining creates a new checkpoint and does not overwrite the deposited one.

## Expected files and troubleshooting

- `01_benchmark/outputs/model_answers.jsonl`: full generated responses and exact prompts.
- `01_benchmark/outputs/model_answers_failures.jsonl`: empty or truncated calls, if any; increase the common token cap and restart the full matrix under a new run label before scoring.
- `01_benchmark/outputs/blind_responses.csv` and `expert_scores_template.csv`: scoring handoff.
- `01_benchmark/outputs/summary/`: expert summary and paired comparisons after scoring.
- `results/cganet_mixed_1600/`: external predictions and metrics.

If `freeze_retrieval.py` reports a missing DOI, complete the DOI mapping before evaluation. If the benchmark reports a missing context, freeze all question contexts first. If the local adapter path is missing, obtain the released adapter or train and label a new model; do not substitute a base-model response for RAMAD. If a requested API model ID is unavailable, record the provider response and revise the preregistered model panel before generating any candidate responses.
