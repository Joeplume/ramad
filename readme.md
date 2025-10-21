# RAMAD: A Large Language Model Framework for Accelerating SERS-Based Detection of Aquaculture Drug Residues

> RAMAD (Raman-enhanced Aquaculture Drug model) is a domain-specific LLM + RAG framework that accelerates the full SERS workflow—substrate selection, multi-drug classification, and quantification—using physics-constrained Nonlinear Mixup (NLMixup) and a Convolution–Global Attention Network (CGANet).

## Highlights
- **RAG 文献库（7,657 篇）**：自动整合知识与建模指引。
- **Physics-constrained NLMixup**：非线性混合增强，CNN 误差降至 ~0.2。
- **CGANet**：多药物分类与定量，单药 RMSE ≈ **0.05**，混合样本 RMSE ≈ **0.2**。
- **鲁棒性**：低浓度与混合样本条件下表现稳定。
- **可扩展**：适用于更广谱的光谱分析、食品安全与环境监测。

## Quickstart
```bash
# 1) Install
pip install -e .

# 2) Prepare corpus index (metadata only; see data/README.md)
bash scripts/download_corpus.sh

# 3) Run end-to-end detection
python -m ramad.workflows.detect --config configs/train.yaml --input data/samples/

# 4) Quantification
python -m ramad.workflows.quantify --model cganet --ckpt path/to/ckpt.pt
