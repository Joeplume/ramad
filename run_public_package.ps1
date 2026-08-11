$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

python (Join-Path $root "04_training_corpus_1000\inspect_dataset.py") (Join-Path $root "04_training_corpus_1000\RAMAD_domain_QA_1000.jsonl")
python (Join-Path $root "04_training_corpus_followup\inspect_examples.py")
python (Join-Path $root "02_code_entrypoints\spectral\inspect_public_spectra.py")
python (Join-Path $root "02_code_entrypoints\ramad_model\train_sft_lora.py") --help
python (Join-Path $root "02_code_entrypoints\ramad_model\run_inference.py") --help
python (Join-Path $root "02_code_entrypoints\ramad_rag\build_index.py") --help
python (Join-Path $root "02_code_entrypoints\ramad_rag\rag_qa.py") --help
python (Join-Path $root "benchmark\run_benchmark.py") --help
python (Join-Path $root "benchmark\aggregate_scores.py") --help

Write-Output "Public package checks completed."
