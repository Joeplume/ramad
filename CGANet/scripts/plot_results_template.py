"""
Template script for plotting regression & classification results.

This is adapted from our internal plotting utilities but simplified:
  - Uses generic CSV / NumPy arrays for true vs. predicted values.
  - Only demonstrates scatter plot + confusion matrix.
"""

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_regression_scatter(
    y_true_internal: Sequence[float],
    y_pred_internal: Sequence[float],
    y_true_external: Sequence[float],
    y_pred_external: Sequence[float],
    y_true_beyond: Sequence[float] | None = None,
    y_pred_beyond: Sequence[float] | None = None,
    save_path: str = "scatter_prediction_template.png",
):
    """
    Scatter plot for internal / external / beyond-limit regression results.

    Parameters
    ----------
    y_true_internal, y_pred_internal : internal test set true/predicted values
    y_true_external, y_pred_external : external test set true/predicted values
    y_true_beyond, y_pred_beyond     : optional beyond-limit validation values
    save_path                        : path to save the PNG figure
    """
    y_true_internal = np.asarray(y_true_internal)
    y_pred_internal = np.asarray(y_pred_internal)
    y_true_external = np.asarray(y_true_external)
    y_pred_external = np.asarray(y_pred_external)

    plt.figure(figsize=(5, 4))

    plt.scatter(
        y_true_internal,
        y_pred_internal,
        alpha=0.5,
        s=30,
        color="#1f77b4",
        label="Internal Test",
        edgecolors="none",
        zorder=1,
    )
    plt.scatter(
        y_true_external,
        y_pred_external,
        alpha=0.9,
        s=30,
        color="#8000ff",
        label="External Test",
        edgecolors="none",
        zorder=2,
    )

    if y_true_beyond is not None and y_pred_beyond is not None:
        y_true_beyond = np.asarray(y_true_beyond)
        y_pred_beyond = np.asarray(y_pred_beyond)
        plt.scatter(
            y_true_beyond,
            y_pred_beyond,
            alpha=0.9,
            s=35,
            color="#ED7D31",
            label="External Test II (Beyond-Limit)",
            edgecolors="none",
            zorder=3,
        )

    # Diagonal line range
    values = [
        y_true_internal,
        y_pred_internal,
        y_true_external,
        y_pred_external,
    ]
    if y_true_beyond is not None and y_pred_beyond is not None:
        values.extend([y_true_beyond, y_pred_beyond])

    min_val = min(v.min() for v in values)
    max_val = max(v.max() for v in values)

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        linewidth=1,
        label="Perfect Prediction",
        zorder=0,
    )

    plt.xlabel("True Concentration")
    plt.ylabel("Predicted Concentration")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(
    y_true_internal: Sequence[int],
    y_pred_internal: Sequence[int],
    y_true_external: Sequence[int],
    y_pred_external: Sequence[int],
    class_names: Sequence[str],
    out_dir: str = "plots_confusion_template",
):
    """
    Plot internal & external confusion matrices.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Internal
    cm_internal = confusion_matrix(y_true_internal, y_pred_internal)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_internal,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "internal_confusion_matrix_template.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # External
    cm_external = confusion_matrix(y_true_external, y_pred_external)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm_external,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "external_confusion_matrix_template.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # This is just a template showing how the functions can be used.
    raise SystemExit("This is a plotting template; import and use the functions instead.")




