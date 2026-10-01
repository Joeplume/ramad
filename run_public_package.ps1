$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-CheckedPython {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python entry-point check failed with exit code $LASTEXITCODE"
    }
}

Invoke-CheckedPython (Join-Path $root "src\ramad_model\inspect_dataset.py") (Join-Path $root "..\Zenodo模型与数据\training\RAMAD_domain_QA_1000.jsonl")
Invoke-CheckedPython (Join-Path $root "training_examples\inspect_examples.py")
Invoke-CheckedPython (Join-Path $root "src\spectral\inspect_public_spectra.py")
Invoke-CheckedPython (Join-Path $root "src\ramad_model\train_sft_lora.py") --help
Invoke-CheckedPython (Join-Path $root "src\ramad_model\run_inference.py") --help
Invoke-CheckedPython (Join-Path $root "src\ramad_rag\build_index.py") --help
Invoke-CheckedPython (Join-Path $root "src\ramad_rag\rag_qa.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\run_benchmark.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\aggregate_scores.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\freeze_retrieval.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\human_primary.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\build_joint_review.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\replay_historical_review.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\summarize_joint_review.py") --help
Invoke-CheckedPython (Join-Path $root "src\spectral\prepare_mixed_1600.py") --help
Invoke-CheckedPython (Join-Path $root "src\spectral\run_checkpoint.py") --help
Invoke-CheckedPython (Join-Path $root "src\spectral\train_mixed_1600.py") --help
Invoke-CheckedPython (Join-Path $root "src\spectral\evaluate_archived_1800.py") --help

Write-Output "Public entry-point checks completed."
