#!/usr/bin/env python3
"""
Training script for Turbofan RUL 1D-CNN Regression Model.

Features:
- Leakage-safe engine-level train/validation splitting.
- Scaler fitting strictly on training engines.
- Early stopping and best-model checkpointing.
- Training history and metadata persistence.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import keras

from src.rul_model import build_dcnn_model
from src.rul_preprocessing import (
    DEFAULT_RUL_CLIP,
    DEFAULT_SEQUENCE_LENGTH,
    prepare_train_val_pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_dcnn")


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 1D-CNN model for Turbofan Remaining Useful Life prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-input",
        type=str,
        default="Train_data.csv",
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/training",
        help="Directory to save model checkpoints and training history.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of engines reserved for validation (0.0 to 1.0).",
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
        help="Threshold cycle for piecewise-linear RUL clipping.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for gradient updates.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate for RMSprop optimizer.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args(args)


def set_seed(seed: int) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def detect_hardware() -> str:
    """Detect available CPU / GPU devices and log environment details."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        device_desc = f"GPU detected: {len(gpus)} device(s) ({gpus[0].name})"
    else:
        device_desc = "CPU execution (TensorFlow CPU / AVX enabled)"
    logger.info(f"Hardware status: {device_desc}")
    return device_desc


def main(args=None) -> int:
    parsed = parse_args(args)

    print("==================================================")
    print("  Turbofan RUL 1D-CNN Model Training Pipeline")
    print("==================================================")
    print(f"  Input Dataset   : {parsed.data_input}")
    print(f"  Output Directory: {parsed.output_dir}")
    print(f"  Validation Ratio: {parsed.val_ratio * 100:.1f}% of engines")
    print(f"  Sequence Length : {parsed.sequence_length}")
    print(f"  RUL Clip Limit  : {parsed.rul_clip}")
    print(f"  Epochs          : {parsed.epochs}")
    print(f"  Batch Size      : {parsed.batch_size}")
    print(f"  Learning Rate   : {parsed.learning_rate}")
    print(f"  Early Stopping  : {parsed.patience} epochs patience")
    print(f"  Random Seed     : {parsed.seed}")
    print("--------------------------------------------------")

    data_path = Path(parsed.data_input)
    if not data_path.is_file():
        sys.stderr.write(f"Error: Dataset not found at '{parsed.data_input}'\n")
        return 1

    if parsed.epochs <= 0 or parsed.batch_size <= 0:
        sys.stderr.write("Error: --epochs and --batch-size must be positive integers.\n")
        return 1

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(parsed.seed)
    hardware_info = detect_hardware()

    try:
        # 1. Prepare data with zero leakage
        logger.info("[1/4] Preparing data: engine-level splitting and normalization...")
        data_bundle = prepare_train_val_pipeline(
            train_path=data_path,
            val_ratio=parsed.val_ratio,
            sequence_length=parsed.sequence_length,
            rul_clip=parsed.rul_clip,
            seed=parsed.seed,
        )

        X_train = data_bundle["X_train"]
        y_train = data_bundle["y_train"]
        X_val = data_bundle["X_val"]
        y_val = data_bundle["y_val"]
        scaler = data_bundle["scaler"]

        train_engines = data_bundle["train_engine_ids"]
        val_engines = data_bundle["val_engine_ids"]
        logger.info(
            f"      -> Train engines: {len(train_engines)}, Samples: {X_train.shape[0]} (shape: {X_train.shape})"
        )
        logger.info(
            f"      -> Val engines  : {len(val_engines)}, Samples: {X_val.shape[0]} (shape: {X_val.shape})"
        )

        # 2. Build model
        logger.info("[2/4] Building 1D-CNN regression architecture...")
        model = build_dcnn_model(
            sequence_length=parsed.sequence_length,
            num_features=X_train.shape[2],
            learning_rate=parsed.learning_rate,
        )

        # 3. Setup callbacks
        checkpoint_path = out_dir / "best_model.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=parsed.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ]

        # 4. Train model
        logger.info("[3/4] Starting training loop...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=parsed.epochs,
            batch_size=parsed.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # 5. Persist artifacts & metadata
        logger.info("[4/4] Saving model artifacts and training metadata...")
        # Save final model state if checkpoint didn't trigger
        if not checkpoint_path.exists():
            model.save(checkpoint_path)

        # Save history as JSON
        hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        hist_path = out_dir / "history.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(hist_dict, f, indent=2)

        # Compute final best metrics
        best_epoch = int(np.argmin(hist_dict["val_loss"])) + 1
        best_val_loss = float(np.min(hist_dict["val_loss"]))
        best_val_mae = float(hist_dict["val_mae"][best_epoch - 1])
        best_val_rmse = float(hist_dict["val_rmse"][best_epoch - 1])

        summary = {
            "dataset": str(data_path),
            "hardware": hardware_info,
            "train_engines_count": len(train_engines),
            "val_engines_count": len(val_engines),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "sequence_length": parsed.sequence_length,
            "num_features": int(X_train.shape[2]),
            "epochs_trained": len(hist_dict["loss"]),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "best_val_rmse": best_val_rmse,
            "train_engine_ids": [int(i) for i in train_engines],
            "val_engine_ids": [int(i) for i in val_engines],
        }
        summary_path = out_dir / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save scaler parameters for standalone inference
        scaler_info = {
            "feature_cols": data_bundle["feature_cols"],
            "data_min": scaler.data_min_.tolist(),
            "data_max": scaler.data_max_.tolist(),
            "scale": scaler.scale_.tolist(),
            "min": scaler.min_.tolist(),
        }
        with open(out_dir / "scaler_params.json", "w", encoding="utf-8") as f:
            json.dump(scaler_info, f, indent=2)

        print("\n==================================================")
        print("  Training Completed Successfully!")
        print(f"  Best Epoch       : {best_epoch} / {len(hist_dict['loss'])}")
        print(f"  Best Val Loss    : {best_val_loss:.4f} (Huber)")
        print(f"  Best Val MAE     : {best_val_mae:.2f} cycles")
        print(f"  Best Val RMSE    : {best_val_rmse:.2f} cycles")
        print(f"  Saved Model      : {checkpoint_path}")
        print(f"  Training Summary : {summary_path}")
        print("==================================================")
        return 0

    except Exception as exc:
        sys.stderr.write(f"Training Failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
