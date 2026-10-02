$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Invoke-CheckedPython {
    & python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python entry-point check failed with exit code $LASTEXITCODE"
    }
}

Invoke-CheckedPython (Join-Path $root "src\ramad_model\inspect_dataset.py") (Join-Path $root "..\ramad_zenodo_package\training\RAMAD_domain_QA_1000.jsonl")
Invoke-CheckedPython (Join-Path $root "training_examples\inspect_examples.py")
Invoke-CheckedPython (Join-Path $root "src\ramad_model\train_sft_lora.py") --help
Invoke-CheckedPython (Join-Path $root "src\ramad_model\run_inference.py") --help
Invoke-CheckedPython (Join-Path $root "src\ramad_rag\build_index.py") --help
Invoke-CheckedPython (Join-Path $root "src\ramad_rag\rag_qa.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\run_benchmark.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\aggregate_scores.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\frontier\run_frontier_eval.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\frontier\aggregate_scores.py") --help
Invoke-CheckedPython (Join-Path $root "interactive_scoring.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\freeze_retrieval.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\human_primary.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\build_joint_review.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\replay_historical_review.py") --help
Invoke-CheckedPython (Join-Path $root "benchmark\summarize_joint_review.py") --help

Invoke-CheckedPython (Join-Path $root "src\spectral\run_checkpoint.py") --help
Invoke-CheckedPython (Join-Path $root "src\spectral\train_cganet.py") --help

Write-Output "Public entry-point checks completed."
