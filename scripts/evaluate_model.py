#!/usr/bin/env python3
"""
Model evaluation and diagnostic visualization script for Turbofan RUL.

Generates:
- Quantitative metrics: MAE, RMSE, R^2, and NASA C-MAPSS PHM Asymmetric Score.
- Error residual statistics (mean, median, standard deviation).
- Diagnostic plots: Actual vs Predicted, Residual Distribution, and Training Curves.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

# Non-interactive backend suitable for headless servers & CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rul_model import load_trained_model
from src.rul_preprocessing import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_RUL_CLIP,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TARGET_COL,
    compute_piecewise_rul,
    generate_sequences,
    load_csv_data,
    scale_sensor_features,
    split_by_engine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_model")


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained Turbofan 1D-CNN regression model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="artifacts/training/best_model.keras",
        help="Path to trained .keras model file.",
    )
    parser.add_argument(
        "--data-input",
        type=str,
        default="Train_data.csv",
        help="Path to evaluation CSV file.",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="artifacts/training/history.json",
        help="Optional path to training history JSON to plot learning curves.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation",
        help="Directory to save evaluation plots and metrics.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio used during training (to evaluate on held-out engines).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed matching the training split.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help="Sliding window size in operational cycles.",
    )
    parser.add_argument(
        "--rul-clip",
        type=int,
        default=DEFAULT_RUL_CLIP,
        help="Maximum threshold cycle for piecewise-linear RUL.",
    )
    parser.add_argument(
        "--engine-id",
        type=int,
        default=None,
        help="Optional specific engine ID to isolate for detailed trajectory plotting.",
    )
    return parser.parse_args(args)


def calculate_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute official NASA C-MAPSS asymmetric scoring function:
        d_i = y_pred - y_true
        s_i = exp(-d_i / 13) - 1   if d_i < 0 (early prediction: less severe)
        s_i = exp(d_i / 10) - 1    if d_i >= 0 (late prediction: severe failure penalty)
    """
    diff = y_pred.flatten() - y_true.flatten()
    scores = np.where(diff < 0, np.exp(-diff / 13.0) - 1.0, np.exp(diff / 10.0) - 1.0)
    return float(np.sum(scores))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive regression and residual metrics."""
    yt = y_true.flatten()
    yp = y_pred.flatten()
    residuals = yt - yp  # Positive = model underpredicted, Negative = model overpredicted

    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))

    # Coefficient of determination R^2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    nasa_score = calculate_nasa_score(yt, yp)

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "nasa_score": round(nasa_score, 2),
        "mean_residual": round(float(np.mean(residuals)), 4),
        "std_residual": round(float(np.std(residuals)), 4),
        "median_residual": round(float(np.median(residuals)), 4),
        "num_samples": int(len(yt)),
    }


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """Plot ground truth RUL versus model predictions."""
    plt.figure(figsize=(10, 5))
    indices = np.arange(len(y_true))
    plt.plot(indices, y_true.flatten(), label="Actual RUL (Ground Truth)", color="#1f77b4", linewidth=2.0)
    plt.plot(indices, y_pred.flatten(), label="Predicted RUL (1D-CNN)", color="#d62728", linestyle="--", linewidth=1.8)
    plt.xlabel("Sample Index (Consecutive Windows)", fontsize=11)
    plt.ylabel("Remaining Useful Life (Cycles)", fontsize=11)
    plt.title(f"Turbofan RUL: Actual vs Predicted {title_suffix}".strip(), fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Plot distribution of prediction residuals."""
    residuals = y_true.flatten() - y_pred.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Residuals Histogram
    axes[0].hist(residuals, bins=35, color="#2ca02c", edgecolor="black", alpha=0.7)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    axes[0].axvline(float(np.mean(residuals)), color="blue", linestyle=":", label=f"Mean: {np.mean(residuals):.2f}")
    axes[0].set_xlabel("Residual (Actual - Predicted)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Prediction Error Distribution", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[0].legend()

    # Scatter of Actual vs Residuals
    axes[1].scatter(y_true.flatten(), residuals, alpha=0.3, color="#9467bd", edgecolors="none")
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Actual RUL (Cycles)", fontsize=11)
    axes[1].set_ylabel("Residual (Cycles)", fontsize=11)
    axes[1].set_title("Residuals vs Actual RUL", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_learning_curves(history_path: Path, output_path: Path) -> None:
    """Plot training and validation loss and RMSE progression."""
    if not history_path.is_file():
        logger.warning(f"History file not found at {history_path}, skipping learning curve plot.")
        return

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss Curves
    axes[0].plot(epochs, history["loss"], label="Train Loss", color="#1f77b4", linewidth=1.8)
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linestyle="--", linewidth=1.8)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Huber Loss", fontsize=11)
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # RMSE Curves
    if "rmse" in history:
        axes[1].plot(epochs, history["rmse"], label="Train RMSE", color="#2ca02c", linewidth=1.8)
    if "val_rmse" in history:
        axes[1].plot(epochs, history["val_rmse"], label="Val RMSE", color="#d62728", linestyle="--", linewidth=1.8)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("RMSE (Cycles)", fontsize=11)
    axes[1].set_title("Root Mean Squared Error Progression", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main(args=None) -> int:
    parsed = parse_args(args)

    print("==================================================")
    print("  Turbofan RUL 1D-CNN Model Evaluation")
    print("==================================================")
    print(f"  Model Path      : {parsed.model_path}")
    print(f"  Data Input      : {parsed.data_input}")
    print(f"  Output Directory: {parsed.output_dir}")
    print("--------------------------------------------------")

    model_path = Path(parsed.model_path)
    if not model_path.is_file():
        sys.stderr.write(f"Error: Model file not found at '{parsed.model_path}'\n")
        return 1

    data_path = Path(parsed.data_input)
    if not data_path.is_file():
        sys.stderr.write(f"Error: Data file not found at '{parsed.data_input}'\n")
        return 1

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Load model
        logger.info(f"[1/4] Loading trained model from {model_path}...")
        model = load_trained_model(model_path)

        # 2. Prepare evaluation data
        logger.info("[2/4] Preparing evaluation dataset...")
        raw_df = load_csv_data(data_path)
        if DEFAULT_TARGET_COL not in raw_df.columns:
            raw_df = compute_piecewise_rul(raw_df, max_cycle=parsed.rul_clip)

        # If evaluating on Train_data.csv, hold out the identical validation engines used in training
        if raw_df["id"].nunique() > 1 and parsed.engine_id is None:
            train_sub, eval_df = split_by_engine(raw_df, val_ratio=parsed.val_ratio, seed=parsed.seed)
            logger.info(f"      -> Evaluating on {eval_df['id'].nunique()} held-out validation engines.")
            from src.rul_preprocessing import fit_sensor_scaler
            scaler = fit_sensor_scaler(train_sub, feature_cols=DEFAULT_FEATURE_COLS)
        elif parsed.engine_id is not None:
            eval_df = raw_df[raw_df["id"] == parsed.engine_id].copy()
            if eval_df.empty:
                sys.stderr.write(f"Error: Engine ID {parsed.engine_id} not found in dataset.\n")
                return 1
            logger.info(f"      -> Evaluating specifically on engine #{parsed.engine_id} ({len(eval_df)} cycles).")
            from src.rul_preprocessing import fit_sensor_scaler
            scaler = fit_sensor_scaler(raw_df, feature_cols=DEFAULT_FEATURE_COLS)
        else:
            eval_df = raw_df
            from src.rul_preprocessing import fit_sensor_scaler
            scaler = fit_sensor_scaler(raw_df, feature_cols=DEFAULT_FEATURE_COLS)

        eval_norm = scale_sensor_features(eval_df, scaler, feature_cols=DEFAULT_FEATURE_COLS)
        X_eval, y_true = generate_sequences(
            eval_norm,
            sequence_length=parsed.sequence_length,
            feature_cols=DEFAULT_FEATURE_COLS,
            label_col=DEFAULT_TARGET_COL,
        )

        # 3. Generate predictions
        logger.info(f"[3/4] Running inference on {X_eval.shape[0]} sequences...")
        y_pred = model.predict(X_eval, verbose=0)

        # 4. Compute metrics and generate diagnostic plots
        logger.info("[4/4] Calculating metrics and rendering diagnostic plots...")
        metrics = compute_metrics(y_true, y_pred)

        # Save metrics JSON
        metrics_file = out_dir / "evaluation_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Generate plots
        pred_plot = out_dir / "actual_vs_predicted.png"
        plot_predictions(y_true, y_pred, pred_plot, title_suffix=f"(Engine {parsed.engine_id})" if parsed.engine_id else "")

        residual_plot = out_dir / "residuals_distribution.png"
        plot_residuals(y_true, y_pred, residual_plot)

        if parsed.history_path:
            hist_plot = out_dir / "loss_and_metrics_curves.png"
            plot_learning_curves(Path(parsed.history_path), hist_plot)

        print("\n==================================================")
        print("  Evaluation Results")
        print("==================================================")
        print(f"  Samples Evaluated: {metrics['num_samples']:,}")
        print(f"  MAE              : {metrics['mae']} cycles")
        print(f"  RMSE             : {metrics['rmse']} cycles")
        print(f"  R^2 Score        : {metrics['r2_score']}")
        print(f"  NASA Score       : {metrics['nasa_score']}")
        print(f"  Mean Error       : {metrics['mean_residual']} cycles")
        print(f"  Std Error        : {metrics['std_residual']} cycles")
        print(f"  Metrics File     : {metrics_file}")
        print(f"  Prediction Plot  : {pred_plot}")
        print(f"  Residuals Plot   : {residual_plot}")
        print("==================================================")
        return 0

    except Exception as exc:
        sys.stderr.write(f"Evaluation Failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
