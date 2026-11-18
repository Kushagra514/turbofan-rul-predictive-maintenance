#!/usr/bin/env python3
"""
Command-Line Interface for NASA C-MAPSS Turbofan RUL Data Preparation.

Usage example:
    python scripts/prepare_data.py \
        --train-input Train_data.csv \
        --test-input test_engine_21.csv \
        --output-dir data/processed
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rul_preprocessing import (
    DEFAULT_RUL_CLIP,
    DEFAULT_SEQUENCE_LENGTH,
    process_pipeline,
    save_processed_outputs,
)


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess NASA C-MAPSS dataset for Remaining Useful Life (RUL) prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-input",
        type=str,
        required=True,
        help="Path to training CSV file (e.g. Train_data.csv).",
    )
    parser.add_argument(
        "--test-input",
        type=str,
        default=None,
        help="Optional path to test CSV file (e.g. test_engine_21.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/processed",
        help="Directory where processed CSVs and numpy tensors will be saved.",
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
        help="Threshold cycle for piecewise-linear RUL clipping (R_early).",
    )
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)

    print("==================================================")
    print("  NASA C-MAPSS Turbofan RUL Data Preparation")
    print("==================================================")
    print(f"  Training Input  : {parsed.train_input}")
    print(f"  Test Input      : {parsed.test_input or '(None)'}")
    print(f"  Output Directory: {parsed.output_dir}")
    print(f"  Sequence Length : {parsed.sequence_length}")
    print(f"  RUL Clip Limit  : {parsed.rul_clip}")
    print("--------------------------------------------------")

    train_path = Path(parsed.train_input)
    if not train_path.exists():
        sys.stderr.write(f"Error: Training file not found at '{parsed.train_input}'\n")
        return 1

    if parsed.test_input:
        test_path = Path(parsed.test_input)
        if not test_path.exists():
            sys.stderr.write(f"Error: Test file not found at '{parsed.test_input}'\n")
            return 1
    else:
        test_path = None

    if parsed.sequence_length <= 0:
        sys.stderr.write(
            f"Error: --sequence-length must be a positive integer, got {parsed.sequence_length}\n"
        )
        return 1

    if parsed.rul_clip <= 0:
        sys.stderr.write(f"Error: --rul-clip must be a positive integer, got {parsed.rul_clip}\n")
        return 1

    try:
        print("[1/4] Loading and transforming dataset...")
        result = process_pipeline(
            train_path=train_path,
            test_path=test_path,
            sequence_length=parsed.sequence_length,
            rul_clip=parsed.rul_clip,
        )

        train_df = result["train_df"]
        train_X = result["train_X"]
        train_y = result["train_y"]
        print(f"      -> Train rows: {len(train_df):,}, Engines: {train_df['id'].nunique()}")
        print(f"      -> Generated Train sequences: {train_X.shape}, Labels: {train_y.shape}")

        if result.get("test_df") is not None:
            test_df = result["test_df"]
            test_X = result["test_X"]
            test_y = result["test_y"]
            y_info = f", Labels: {test_y.shape}" if test_y is not None else " (unlabeled)"
            print(f"      -> Test rows: {len(test_df):,}, Engines: {test_df['id'].nunique()}")
            print(f"      -> Generated Test sequences: {test_X.shape}{y_info}")

        print("[2/4] Saving processed artifacts to output directory...")
        saved = save_processed_outputs(parsed.output_dir, result)

        print("[3/4] Successfully generated artifacts:")
        for name, p in saved.items():
            size_kb = p.stat().st_size / 1024
            print(f"      * {p.name:25} ({size_kb:8.1f} KB) -> {p}")

        print("[4/4] Done. Data is prepared for model training.")
        print("==================================================")
        return 0

    except Exception as exc:
        sys.stderr.write(f"Pipeline Execution Failed: {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
