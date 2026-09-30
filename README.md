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
| `prompts/` | The exact prompts used by the new comparison and the recovered original RAG QA template |

The original FAISS index was recovered from the author's backup. It contains 1,008,839 passages from 7,645 distinct source filenames. The backup also contains 7,551 PDFs; these counts refer to indexed sources and surviving PDFs, respectively, while the paper reports 7,657 originally collected publications. The PDFs and full-text index are not included in this source repository because source permissions must be checked. The five frozen benchmark contexts are included in `01_benchmark/retrieval_contexts.jsonl`. Their source metadata identifies document titles but contains no page numbers; DOIs are included only when independently verified. The original Qwen3-4B LoRA adapter is still missing; training a replacement creates a newly labeled model. Model artifacts and spectral data are prepared separately for a versioned Zenodo deposit. The earlier Zenodo record, [10.5281/zenodo.17809450](https://doi.org/10.5281/zenodo.17809450), contains one file named `model-00001-of-00004.safetensors` and is not an executable adapter package.

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

1. Use the recovered original FAISS index with the matching `all-MiniLM-L6-v2` encoder, or rebuild an index from an authorized PDF set with `02_code_entrypoints/ramad_rag/build_index.py`. The original script split text into 500-character chunks with 100-character overlap; the index and top-five retrieval settings are recorded in `01_benchmark/retrieval_config.json`.
2. Run `01_benchmark/freeze_retrieval.py` to write one fixed `retrieval_contexts.jsonl` record per question. If verified DOIs are available, supply a mapping CSV with `--doi-map`; missing DOIs and page numbers are left blank. Use the same five passages for every `prompt_rag` candidate.
3. Copy `01_benchmark/benchmark_config.example.json` to `01_benchmark/benchmark_config.json`. Replace the RAMAD adapter path with the released directory; a relative path is resolved from the configuration file. `dry-run` checks for adapter configuration and weights before model calls. Verify the exact API model IDs and save the access date and provider in the run record.
4. Run the commands below. The `dry-run` lists the complete model and question matrix before any model call. `generate` records the actual messages, retrieved passages, parameters, response, returned model ID, and usage for each call.
5. Score every blinded answer with the four reviewer model families and the same six dimensions used in the paper. Save raw reviewer outputs, then run `aggregate_scores.py` for arithmetic-mean and original-style iterative-weight summaries. Independent human blind scores may be added as a separate quality check with `human_primary.py`.

```powershell
python 01_benchmark/freeze_retrieval.py --index-dir data/faiss_index --allow-pickle
python 01_benchmark/run_benchmark.py dry-run --config benchmark_config.json
python 01_benchmark/run_benchmark.py generate --config benchmark_config.json
python 01_benchmark/run_benchmark.py score --config benchmark_config.json
python 01_benchmark/aggregate_scores.py --config benchmark_config.json --scores outputs/model_scores.jsonl
```

The five existing questions are in `01_benchmark/questions.jsonl`. The common conditions, six scoring dimensions, and analysis plan are specified in `01_benchmark/BENCHMARK_PROTOCOL.md`. Use one frozen run label and one scoring round for all candidates. Do not combine historical RAMAD scores with newly generated frontier-model scores. The complete answer and reviewer-scoring matrix is required before reporting a model ranking.

The completed seven-model comparison, using one frozen set of retrieved passages and prompts for each question, is recorded in `01_benchmark/outputs/`. Under the paper's six-dimension, four-reviewer iterative weighting method, the newly reconstructed RAMAD adapter scored 22.737/30, the untuned Qwen3-4B backbone 23.709/30, Claude Fable 5 28.962/30, and Kimi K3 28.721/30. The reconstructed adapter is a new model trained from the archived instruction corpus, not the missing historical adapter. These scores therefore describe the released reconstruction and do not replace the paper's historical RAMAD score. The answer log, raw reviewer scores, per-question comparisons, and arithmetic-mean sensitivity summary are included. Five questions are insufficient for a definitive significance claim.

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

If the benchmark reports a missing context, freeze all question contexts first. If the local adapter path is missing, obtain the original adapter or train and label a new model. If a requested API model ID is unavailable, record the provider response and revise the preregistered model panel before generating any candidate responses.
