"""
Unit tests for RUL preprocessing module and data preparation CLI.
Uses small synthetic datasets and temporary files (no NASA dataset dependency).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_data import main as cli_main
from src.rul_preprocessing import (
    DEFAULT_FEATURE_COLS,
    DEFAULT_RUL_CLIP,
    DEFAULT_SEQUENCE_LENGTH,
    compute_piecewise_rul,
    fit_sensor_scaler,
    generate_sequences,
    load_csv_data,
    process_pipeline,
    save_processed_outputs,
    scale_sensor_features,
)


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """
    Generate synthetic multi-engine dataset with known cycle counts and features.
    Engine 1: 150 cycles (exceeds 125 RUL clip and 40 sequence length).
    Engine 2: 60 cycles (exceeds 40 sequence length, does not reach 125).
    Engine 3: 20 cycles (fewer than sequence length 40, should be safely skipped).
    """
    rows = []
    configs = [(1, 150), (2, 60), (3, 20)]
    for eng_id, total_cycles in configs:
        for c in range(1, total_cycles + 1):
            row = {"id": eng_id, "cycle": c}
            for col in DEFAULT_FEATURE_COLS:
                # Deterministic synthetic values
                row[col] = float(hash(f"{col}_{eng_id}_{c}") % 100) + float(c)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_test_df() -> pd.DataFrame:
    """Generate synthetic test engine with 70 cycles."""
    rows = []
    for c in range(1, 71):
        row = {"id": 21, "cycle": c}
        for col in DEFAULT_FEATURE_COLS:
            row[col] = float(c * 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Data Loading & Validation Tests
# ---------------------------------------------------------------------------


def test_load_csv_data_valid(tmp_path: Path, synthetic_train_df: pd.DataFrame):
    csv_file = tmp_path / "valid.csv"
    synthetic_train_df.to_csv(csv_file, index=False)

    loaded = load_csv_data(csv_file, required_cols=["id", "cycle"])
    assert len(loaded) == len(synthetic_train_df)
    assert list(loaded.columns) == list(synthetic_train_df.columns)


def test_load_csv_data_missing_file():
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_csv_data("non_existent_file_path_12345.csv")


def test_load_csv_data_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_csv_data(empty_file)


def test_load_csv_data_missing_columns(tmp_path: Path):
    csv_file = tmp_path / "incomplete.csv"
    df = pd.DataFrame({"id": [1, 2], "cycle": [1, 2]})
    df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_csv_data(csv_file, required_cols=["id", "cycle", "sensor2", "sensor3"])


# ---------------------------------------------------------------------------
# 2. RUL Calculation Tests
# ---------------------------------------------------------------------------


def test_compute_piecewise_rul():
    df = pd.DataFrame({
        "id": [1] * 150,
        "cycle": list(range(1, 151)),
    })
    result = compute_piecewise_rul(df, max_cycle=125)

    assert "remaining_cycle" in result.columns
    assert "R_early" in result.columns

    # Cycle 1: remaining is 150 - 1 = 149 -> clipped to 125
    assert result.loc[result["cycle"] == 1, "R_early"].iloc[0] == 125
    # Cycle 25: remaining is 150 - 25 = 125 -> exactly 125
    assert result.loc[result["cycle"] == 25, "R_early"].iloc[0] == 125
    # Cycle 50: remaining is 150 - 50 = 100 -> linear degradation 100
    assert result.loc[result["cycle"] == 50, "R_early"].iloc[0] == 100
    # Last cycle: remaining is 0 -> 0
    assert result.loc[result["cycle"] == 150, "R_early"].iloc[0] == 0


def test_compute_piecewise_rul_missing_columns():
    df = pd.DataFrame({"some_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="must be present in DataFrame"):
        compute_piecewise_rul(df)


# ---------------------------------------------------------------------------
# 3. Scaler & Data Leakage Prevention Tests
# ---------------------------------------------------------------------------


def test_scaler_no_data_leakage(synthetic_train_df: pd.DataFrame, synthetic_test_df: pd.DataFrame):
    scaler = fit_sensor_scaler(synthetic_train_df)

    # Transform both
    train_norm = scale_sensor_features(synthetic_train_df, scaler)
    test_norm = scale_sensor_features(synthetic_test_df, scaler)

    # Train feature values must be strictly between 0 and 1
    for col in DEFAULT_FEATURE_COLS:
        assert train_norm[col].min() >= -1e-6
        assert train_norm[col].max() <= 1.0 + 1e-6

    # Test features transformed with train statistics: metadata columns untouched
    assert list(test_norm["id"]) == list(synthetic_test_df["id"])
    assert list(test_norm["cycle"]) == list(synthetic_test_df["cycle"])


# ---------------------------------------------------------------------------
# 4. Sequence & Window Generation Tests
# ---------------------------------------------------------------------------


def test_generate_sequences_shapes_and_alignment(synthetic_train_df: pd.DataFrame):
    df_with_rul = compute_piecewise_rul(synthetic_train_df, max_cycle=125)
    seq_len = 40

    X, y = generate_sequences(df_with_rul, sequence_length=seq_len)

    # Engine 1: 150 cycles -> 150 - 40 = 110 windows
    # Engine 2: 60 cycles  -> 60 - 40  = 20 windows
    # Engine 3: 20 cycles  -> < 40     = 0 windows (safely skipped)
    expected_samples = (150 - 40) + (60 - 40)
    assert X.shape == (expected_samples, seq_len, len(DEFAULT_FEATURE_COLS))
    assert y.shape == (expected_samples, 1)

    # Check temporal alignment for Engine 1 first window
    # Window index 0: cycles 1 to 40 (rows 0 to 39)
    # Target index 0: cycle 41 (row 40)
    eng1_df = df_with_rul[df_with_rul["id"] == 1].reset_index(drop=True)
    expected_y0 = eng1_df.loc[seq_len, "R_early"]
    assert np.isclose(y[0, 0], expected_y0)


def test_generate_sequences_invalid_sequence_length(synthetic_train_df: pd.DataFrame):
    with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
        generate_sequences(synthetic_train_df, sequence_length=0)

    with pytest.raises(ValueError, match="Cannot generate any sequence samples"):
        generate_sequences(synthetic_train_df, sequence_length=500)


# ---------------------------------------------------------------------------
# 5. Determinism & End-to-End Pipeline Tests
# ---------------------------------------------------------------------------


def test_process_pipeline_deterministic(tmp_path: Path, synthetic_train_df: pd.DataFrame, synthetic_test_df: pd.DataFrame):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    synthetic_train_df.to_csv(train_file, index=False)
    synthetic_test_df.to_csv(test_file, index=False)

    run1 = process_pipeline(train_file, test_file, sequence_length=40, rul_clip=125)
    run2 = process_pipeline(train_file, test_file, sequence_length=40, rul_clip=125)

    assert np.array_equal(run1["train_X"], run2["train_X"])
    assert np.array_equal(run1["train_y"], run2["train_y"])
    assert np.array_equal(run1["test_X"], run2["test_X"])
    pd.testing.assert_frame_equal(run1["train_df"], run2["train_df"])


def test_save_processed_outputs_creates_directory_and_artifacts(
    tmp_path: Path, synthetic_train_df: pd.DataFrame, synthetic_test_df: pd.DataFrame
):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    synthetic_train_df.to_csv(train_file, index=False)
    synthetic_test_df.to_csv(test_file, index=False)

    result = process_pipeline(train_file, test_file, sequence_length=40, rul_clip=125)
    output_dir = tmp_path / "nested" / "processed_data"

    saved = save_processed_outputs(output_dir, result)

    assert output_dir.is_dir()
    assert (output_dir / "train_normalized.csv").exists()
    assert (output_dir / "test_normalized.csv").exists()
    assert (output_dir / "train_sequences.npz").exists()
    assert (output_dir / "test_sequences.npz").exists()
    assert (output_dir / "metadata.json").exists()

    with open(output_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["train_engines"] == 3
    assert meta["input_shape"] == [40, 14]


# ---------------------------------------------------------------------------
# 6. Command-Line Interface (CLI) Smoke Tests
# ---------------------------------------------------------------------------


def test_cli_success(tmp_path: Path, synthetic_train_df: pd.DataFrame, synthetic_test_df: pd.DataFrame):
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"
    synthetic_train_df.to_csv(train_file, index=False)
    synthetic_test_df.to_csv(test_file, index=False)
    out_dir = tmp_path / "cli_out"

    exit_code = cli_main([
        "--train-input", str(train_file),
        "--test-input", str(test_file),
        "--output-dir", str(out_dir),
        "--sequence-length", "40",
        "--rul-clip", "125",
    ])
    assert exit_code == 0
    assert (out_dir / "train_sequences.npz").exists()
    assert (out_dir / "train_normalized.csv").exists()


def test_cli_missing_file_returns_error(tmp_path: Path):
    exit_code = cli_main([
        "--train-input", str(tmp_path / "does_not_exist.csv"),
    ])
    assert exit_code == 1


def test_cli_invalid_arguments(tmp_path: Path, synthetic_train_df: pd.DataFrame):
    train_file = tmp_path / "train.csv"
    synthetic_train_df.to_csv(train_file, index=False)

    exit_code = cli_main([
        "--train-input", str(train_file),
        "--sequence-length", "-5",
    ])
    assert exit_code == 1
