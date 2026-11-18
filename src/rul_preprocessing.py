"""
RUL Preprocessing Module for NASA C-MAPSS Turbofan Engine Dataset.

Provides reusable, testable routines for:
1. Loading and validating raw or cleaned C-MAPSS data.
2. Computing piecewise-linear Remaining Useful Life (RUL) targets (R_early).
3. Normalizing sensor features without test-set data leakage.
4. Generating temporal sliding-window sequences for 1D-CNN regression.
5. Persisting processed tabular and tensor artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# Default sensor columns selected in upstream research for C-MAPSS FD001
DEFAULT_FEATURE_COLS: List[str] = [
    "sensor2",
    "sensor3",
    "sensor4",
    "sensor7",
    "sensor8",
    "sensor9",
    "sensor11",
    "sensor12",
    "sensor13",
    "sensor14",
    "sensor15",
    "sensor17",
    "sensor20",
    "sensor21",
]

DEFAULT_METADATA_COLS: List[str] = ["id", "cycle"]
DEFAULT_TARGET_COL: str = "R_early"
DEFAULT_SEQUENCE_LENGTH: int = 40
DEFAULT_RUL_CLIP: int = 125

RAW_COLUMNS: List[str] = [
    "id",
    "cycle",
    "op1",
    "op2",
    "op3",
    "sensor1",
    "sensor2",
    "sensor3",
    "sensor4",
    "sensor5",
    "sensor6",
    "sensor7",
    "sensor8",
    "sensor9",
    "sensor10",
    "sensor11",
    "sensor12",
    "sensor13",
    "sensor14",
    "sensor15",
    "sensor16",
    "sensor17",
    "sensor18",
    "sensor19",
    "sensor20",
    "sensor21",
    "sensor22",
    "sensor23",
]


def load_csv_data(
    filepath: Union[str, Path],
    required_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a CSV dataset and validate its structure.

    Args:
        filepath: Path to the CSV file.
        required_cols: Optional list of column names that must be present.

    Returns:
        pd.DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If file is empty or required columns are missing.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"Input file is completely empty: {path}") from err

    if df.empty:
        raise ValueError(f"Loaded DataFrame contains 0 rows: {path}")

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset at {path} is missing required columns: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    return df


def compute_piecewise_rul(
    df: pd.DataFrame,
    max_cycle: int = DEFAULT_RUL_CLIP,
    id_col: str = "id",
    cycle_col: str = "cycle",
    target_col: str = DEFAULT_TARGET_COL,
) -> pd.DataFrame:
    """
    Calculate piecewise-linear Remaining Useful Life (R_early) per engine.

    In the NASA C-MAPSS dataset, engines degrade gradually. Following standard
    practice (Heimes 2008, Ayodeji 2021), RUL is assumed constant at max_cycle
    during early normal operation and degrades linearly thereafter:
        remaining_cycle = max(cycle) - cycle
        R_early = min(remaining_cycle, max_cycle)

    Args:
        df: DataFrame containing engine id and cycle columns.
        max_cycle: Threshold cycle for piecewise-linear RUL clipping (default 125).
        id_col: Name of engine identifier column.
        cycle_col: Name of operational cycle column.
        target_col: Destination column name for clipped RUL.

    Returns:
        New pd.DataFrame with target_col and remaining_cycle added.
    """
    if id_col not in df.columns or cycle_col not in df.columns:
        raise ValueError(f"Columns '{id_col}' and '{cycle_col}' must be present in DataFrame.")

    result = df.copy()
    max_cycles = result.groupby(id_col)[cycle_col].transform("max")
    result["remaining_cycle"] = max_cycles - result[cycle_col]
    result[target_col] = result["remaining_cycle"].apply(
        lambda x: max_cycle if x >= max_cycle else x
    )
    return result


def fit_sensor_scaler(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on specified sensor feature columns from training data.

    Prevents data leakage by computing statistics strictly on the training partition.

    Args:
        train_df: Training DataFrame.
        feature_cols: List of feature column names to scale.

    Returns:
        Fitted sklearn MinMaxScaler instance.
    """
    cols = feature_cols or DEFAULT_FEATURE_COLS
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Training data is missing feature columns for scaling: {missing}")

    scaler = MinMaxScaler()
    scaler.fit(train_df[cols].values)
    return scaler


def scale_sensor_features(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to feature columns of a DataFrame.

    Metadata columns (e.g. id, cycle, R_early) are preserved untouched.

    Args:
        df: DataFrame to transform.
        scaler: Pre-fitted MinMaxScaler.
        feature_cols: Feature columns to scale.

    Returns:
        Transformed pd.DataFrame with scaled feature values.
    """
    cols = feature_cols or DEFAULT_FEATURE_COLS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing feature columns for scaling: {missing}")

    result = df.copy()
    scaled_values = scaler.transform(result[cols].values)
    result[cols] = scaled_values
    return result


def generate_sequences(
    df: pd.DataFrame,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    feature_cols: Optional[List[str]] = None,
    label_col: Optional[str] = DEFAULT_TARGET_COL,
    id_col: str = "id",
    cycle_col: str = "cycle",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Extract sliding-window sequences for 1D-CNN temporal modeling.

    Each window has shape (sequence_length, num_features).
    The corresponding target label is the RUL value at the end of that window.

    Matches the original notebook's sliding window generator:
        window i: rows [i : i + sequence_length]
        label i: row [i + sequence_length]

    Args:
        df: DataFrame sorted by id and cycle.
        sequence_length: Window size in cycles (default 40).
        feature_cols: Features included in each temporal sequence.
        label_col: Name of RUL label column, or None if generating unlabeled sequences.
        id_col: Engine identifier column.
        cycle_col: Operational cycle column for sorting.

    Returns:
        Tuple of (X, y):
            X: np.ndarray of shape (num_samples, sequence_length, num_features), dtype float32.
            y: np.ndarray of shape (num_samples, 1), dtype float32, or None if label_col is None.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive integer, got {sequence_length}")

    cols = feature_cols or DEFAULT_FEATURE_COLS
    has_labels = label_col is not None and label_col in df.columns

    seq_list: List[np.ndarray] = []
    label_list: List[float] = []

    for _, group in df.groupby(id_col):
        # Guarantee strict chronological ordering by cycle within each engine
        if cycle_col in group.columns:
            group = group.sort_values(cycle_col)

        feature_matrix = group[cols].to_numpy(dtype=np.float32)
        num_rows = feature_matrix.shape[0]

        if num_rows <= sequence_length:
            continue

        for i in range(num_rows - sequence_length):
            seq_list.append(feature_matrix[i : i + sequence_length])

        if has_labels:
            target_vector = group[label_col].to_numpy(dtype=np.float32)
            for i in range(num_rows - sequence_length):
                label_list.append(target_vector[i + sequence_length])

    if not seq_list:
        raise ValueError(
            f"No engines have more than sequence_length ({sequence_length}) cycles. "
            "Cannot generate any sequence samples."
        )

    seq_array = np.array(seq_list, dtype=np.float32)
    label_array = np.array(label_list, dtype=np.float32).reshape(-1, 1) if has_labels else None

    return seq_array, label_array


def split_by_engine(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    id_col: str = "id",
    seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into train and validation partitions strictly at the engine level.

    Prevents temporal data leakage: all operational cycles of a given engine
    reside entirely in either the train or validation partition.

    Args:
        df: Tabular DataFrame containing engine records.
        val_ratio: Proportion of unique engines reserved for validation (0.0 to 1.0).
        id_col: Column identifying engines.
        seed: Random seed for reproducible engine selection (or None).

    Returns:
        Tuple of (train_df, val_df).
    """
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be strictly between 0.0 and 1.0, got {val_ratio}")

    engine_ids = np.sort(df[id_col].unique())
    num_engines = len(engine_ids)
    if num_engines < 2:
        raise ValueError(f"Cannot split by engine: dataset contains only {num_engines} engine.")

    if seed is not None:
        rng = np.random.RandomState(seed)
        shuffled_ids = rng.permutation(engine_ids)
    else:
        shuffled_ids = engine_ids.copy()

    num_val = max(1, int(round(num_engines * val_ratio)))
    val_ids = set(shuffled_ids[:num_val])
    train_ids = set(shuffled_ids[num_val:])

    train_df = df[df[id_col].isin(train_ids)].copy()
    val_df = df[df[id_col].isin(val_ids)].copy()
    return train_df, val_df


def prepare_train_val_pipeline(
    train_path: Union[str, Path],
    val_ratio: float = 0.2,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    rul_clip: int = DEFAULT_RUL_CLIP,
    feature_cols: Optional[List[str]] = None,
    seed: Optional[int] = 42,
) -> Dict[str, Union[pd.DataFrame, np.ndarray, MinMaxScaler, List]]:
    """
    End-to-end training and validation data preparation pipeline.

    1. Loads dataset and validates schema.
    2. Computes piecewise-linear RUL targets (R_early) if not already present.
    3. Splits data into train and validation sets strictly by engine ID.
    4. Fits MinMaxScaler exclusively on train engines (zero test/val leakage).
    5. Scales features for both train and validation sets.
    6. Generates 3D sliding-window sequence arrays and aligned labels.

    Returns:
        Dictionary containing X_train, y_train, X_val, y_val, scaler, and engine ID lists.
    """
    cols = feature_cols or DEFAULT_FEATURE_COLS
    required_base = ["id", "cycle"] + cols

    raw_df = load_csv_data(train_path, required_cols=required_base)
    if DEFAULT_TARGET_COL not in raw_df.columns:
        raw_df = compute_piecewise_rul(raw_df, max_cycle=rul_clip)

    train_df, val_df = split_by_engine(raw_df, val_ratio=val_ratio, seed=seed)

    # Fit scaler strictly on train engines
    scaler = fit_sensor_scaler(train_df, feature_cols=cols)

    # Transform both
    train_norm = scale_sensor_features(train_df, scaler, feature_cols=cols)
    val_norm = scale_sensor_features(val_df, scaler, feature_cols=cols)

    # Generate sequences
    X_train, y_train = generate_sequences(
        train_norm,
        sequence_length=sequence_length,
        feature_cols=cols,
        label_col=DEFAULT_TARGET_COL,
    )
    X_val, y_val = generate_sequences(
        val_norm,
        sequence_length=sequence_length,
        feature_cols=cols,
        label_col=DEFAULT_TARGET_COL,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "scaler": scaler,
        "feature_cols": cols,
        "train_engine_ids": sorted(list(train_df["id"].unique())),
        "val_engine_ids": sorted(list(val_df["id"].unique())),
        "sequence_length": sequence_length,
        "rul_clip": rul_clip,
    }


def process_pipeline(
    train_path: Union[str, Path],
    test_path: Optional[Union[str, Path]] = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    rul_clip: int = DEFAULT_RUL_CLIP,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Union[pd.DataFrame, np.ndarray, MinMaxScaler, Dict]]:
    """
    End-to-end data preparation pipeline.

    1. Loads train (and optional test) CSVs.
    2. Calculates piecewise-linear RUL targets if not already present.
    3. Fits MinMaxScaler strictly on training features.
    4. Scales train and test features.
    5. Generates sliding-window sequences and aligned labels.

    Returns:
        Dictionary containing normalized DataFrames, sequence arrays, and fitted scaler.
    """
    cols = feature_cols or DEFAULT_FEATURE_COLS
    required_base = ["id", "cycle"] + cols

    # 1. Load data
    train_df = load_csv_data(train_path, required_cols=required_base)
    test_df = load_csv_data(test_path, required_cols=required_base) if test_path else None

    # 2. Compute RUL if not present
    if DEFAULT_TARGET_COL not in train_df.columns:
        train_df = compute_piecewise_rul(train_df, max_cycle=rul_clip)

    if test_df is not None and DEFAULT_TARGET_COL not in test_df.columns:
        test_df = compute_piecewise_rul(test_df, max_cycle=rul_clip)

    # 3. Fit scaler strictly on train
    scaler = fit_sensor_scaler(train_df, feature_cols=cols)

    # 4. Normalize features
    train_norm_df = scale_sensor_features(train_df, scaler, feature_cols=cols)
    test_norm_df = scale_sensor_features(test_df, scaler, feature_cols=cols) if test_df is not None else None

    # 5. Generate sequences
    train_X, train_y = generate_sequences(
        train_norm_df,
        sequence_length=sequence_length,
        feature_cols=cols,
        label_col=DEFAULT_TARGET_COL,
    )

    test_X, test_y = (None, None)
    if test_norm_df is not None:
        test_X, test_y = generate_sequences(
            test_norm_df,
            sequence_length=sequence_length,
            feature_cols=cols,
            label_col=DEFAULT_TARGET_COL if DEFAULT_TARGET_COL in test_norm_df.columns else None,
        )

    return {
        "train_df": train_norm_df,
        "test_df": test_norm_df,
        "train_X": train_X,
        "train_y": train_y,
        "test_X": test_X,
        "test_y": test_y,
        "scaler": scaler,
        "feature_cols": cols,
        "sequence_length": sequence_length,
        "rul_clip": rul_clip,
    }


def save_processed_outputs(
    output_dir: Union[str, Path],
    pipeline_result: Dict,
) -> Dict[str, Path]:
    """
    Persist normalized CSVs, sequence arrays, and metadata to disk.

    Output files:
    - train_normalized.csv
    - test_normalized.csv (if test data provided)
    - train_sequences.npz (arrays 'X' and 'y')
    - test_sequences.npz (arrays 'X' and optional 'y')
    - metadata.json (summary statistics, column lists, and shapes)

    Args:
        output_dir: Destination directory path.
        pipeline_result: Output dictionary from process_pipeline.

    Returns:
        Dictionary mapping artifact name to its Path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: Dict[str, Path] = {}

    # Save normalized CSVs
    train_csv_path = out_path / "train_normalized.csv"
    pipeline_result["train_df"].to_csv(train_csv_path, index=False)
    saved_files["train_csv"] = train_csv_path

    if pipeline_result.get("test_df") is not None:
        test_csv_path = out_path / "test_normalized.csv"
        pipeline_result["test_df"].to_csv(test_csv_path, index=False)
        saved_files["test_csv"] = test_csv_path

    # Save sequence arrays
    train_npz_path = out_path / "train_sequences.npz"
    np.savez_compressed(
        train_npz_path,
        X=pipeline_result["train_X"],
        y=pipeline_result["train_y"],
    )
    saved_files["train_sequences"] = train_npz_path

    if pipeline_result.get("test_X") is not None:
        test_npz_path = out_path / "test_sequences.npz"
        npz_kwargs = {"X": pipeline_result["test_X"]}
        if pipeline_result.get("test_y") is not None:
            npz_kwargs["y"] = pipeline_result["test_y"]
        np.savez_compressed(test_npz_path, **npz_kwargs)
        saved_files["test_sequences"] = test_npz_path

    # Save metadata JSON
    metadata = {
        "feature_cols": pipeline_result["feature_cols"],
        "sequence_length": pipeline_result["sequence_length"],
        "rul_clip": pipeline_result["rul_clip"],
        "train_rows": len(pipeline_result["train_df"]),
        "train_engines": int(pipeline_result["train_df"]["id"].nunique()),
        "train_samples": int(pipeline_result["train_X"].shape[0]),
        "input_shape": list(pipeline_result["train_X"].shape[1:]),
    }
    if pipeline_result.get("test_df") is not None:
        metadata["test_rows"] = len(pipeline_result["test_df"])
        metadata["test_engines"] = int(pipeline_result["test_df"]["id"].nunique())
        metadata["test_samples"] = int(pipeline_result["test_X"].shape[0])

    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    saved_files["metadata"] = meta_path

    return saved_files
