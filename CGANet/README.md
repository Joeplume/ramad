# CGANet MIC+Transformer (Template Release)

This folder contains a **minimal, publication-oriented template** of the core
code used in our work. It is designed for:

- Showing the overall model architecture (MIC + Transformer fusion),
- Documenting the training and plotting logic at a high level,
- **Without** enabling direct one-click reproduction of our exact experiments.

Some hyper-parameters and paths are intentionally left as **placeholders**.

---

## Structure

- `models/cganet_mic_transformer.py`
  - Contains the `MIC` block and `RamanMICTransformerFusionModel`.
  - The MIC branch uses multi-scale convolutions + isometric conv.
  - The Transformer branch is a standard `nn.TransformerEncoder`.
  - The two branches are fused by a weighted sum.
  - **Important**: the fusion ratio (MIC vs Transformer) is **not fixed**
    here – you must choose your own `alpha`.

- `train/train_cganet_template.py`
  - A **template training loop**:
    - CSV reading,
    - feature scaling with `RobustScaler`,
    - label encoding with `LabelEncoder`,
    - model construction and training loop.
  - Data paths and training hyper-parameters (`lr`, `num_epochs`) are left
    as `...` to be filled in by users.

- `scripts/plot_results_template.py`
  - Minimal plotting utilities:
    - regression scatter for internal / external / beyond-limit,
    - internal & external confusion matrices.
  - Paths and arrays are passed as arguments; this file contains no I/O.

- `requirements.txt`
  - Lists the main Python dependencies used by the templates.

---

## Non-runnable placeholders (on purpose)

To keep the release open and transparent while avoiding blind copy–run:

- In `cganet_mic_transformer.py`:

  - Fusion ratio is left as a placeholder:

    ```python
    # Example (NOT fixed in code):
    # fused = 0.9 * mic_feat + 0.1 * trans_feat
    alpha = ...  # user must choose alpha in [0, 1]
    fused = alpha * mic_feat + (1.0 - alpha) * trans_feat
    ```

- In `train/train_cganet_template.py`:

  - Data paths and key hyper-parameters are placeholders:

    ```python
    train_csv = "/path/to/your/augmented_multi_substances_train.csv"
    lr = ...          # user chooses learning rate
    num_epochs = ...  # user chooses number of epochs
    ```

  - At the bottom, the script exits with a message instead of training
    if run directly, to emphasize it is a template:

    ```python
    if __name__ == "__main__":
        raise SystemExit("This is a template. Please customize it before running.")
    ```

---

## How to use this template

1. **Clone or copy** this folder into your own project / GitHub repository.
2. In `train/train_cganet_template.py`:
   - Set `train_csv` to your own CSV file.
   - Uncomment or adapt feature-length handling if needed.
   - Choose reasonable values for `lr` and `num_epochs`.
3. In `models/cganet_mic_transformer.py`:
   - Decide your own fusion ratio `alpha` between MIC and Transformer.
   - Optionally adjust kernel sizes, hidden dimensions, etc.
4. (Optional) Use the functions in `scripts/plot_results_template.py`
   to visualize your regression and classification results.

This template is intended as a **reference implementation**, not a full,
ready-made training pipeline. Users are encouraged to adapt it to their own
datasets and research questions.



