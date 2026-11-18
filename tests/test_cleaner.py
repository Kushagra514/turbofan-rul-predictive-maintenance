"""
Unit and regression tests for C-MAPSS data cleaning and RUL preprocessing functions.
"""

import pytest
import pandas as pd
import importlib.util
from pathlib import Path

# Load module directly from local file
MODULE_PATH = Path(__file__).resolve().parent.parent / "C-MAPSS_data_cleaner.py"
spec = importlib.util.spec_from_file_location("cleaner", MODULE_PATH)
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)


@pytest.fixture
def sample_train_df():
    """Create a minimal synthetic dataframe matching NASA C-MAPSS structure."""
    data = []
    # Engine 1 runs for 150 cycles, Engine 2 runs for 50 cycles
    for engine_id, max_cycles in [(1, 150), (2, 50)]:
        for c in range(1, max_cycles + 1):
            row = {"id": engine_id, "cycle": c}
            for op in ["op1", "op2", "op3"]:
                row[op] = 0.0
            for s in range(1, 24):
                row[f"sensor{s}"] = float(s * 10 + c)
            data.append(row)
    return pd.DataFrame(data)


@pytest.fixture
def sample_test_df():
    """Create minimal test dataframe and ground-truth RUL results."""
    data = []
    for c in range(1, 31):
        row = {"id": 1, "cycle": c}
        for op in ["op1", "op2", "op3"]:
            row[op] = 0.0
        for s in range(1, 24):
            row[f"sensor{s}"] = float(s * 10 + c)
        data.append(row)
    test_df = pd.DataFrame(data)
    test_results = pd.DataFrame({"rul": [112], "null": [0]})
    return test_results, test_df


def test_train_data_cleaner(sample_train_df):
    cleaned = cleaner.train_data_cleaner(sample_train_df.copy())
    assert "remaining_cycle" in cleaned.columns
    assert "id" not in cleaned.columns
    assert "cycle" not in cleaned.columns
    assert "sensor1" not in cleaned.columns
    assert "sensor2" in cleaned.columns
    # For engine 1 at cycle 1: max cycle 150 -> remaining 149
    assert cleaned.iloc[0]["remaining_cycle"] == 149


def test_train_data_cleaner_R_early_no_name_error(sample_train_df):
    """Verify bugfix: ensure no NameError on df_train and R_early clipping at 125."""
    cleaned = cleaner.train_data_cleaner_R_early(sample_train_df.copy())
    assert "R_early" in cleaned.columns
    assert "id" not in cleaned.columns
    assert "cycle" not in cleaned.columns
    # Cycle 1 has remaining_cycle 149, clipped at 125
    assert cleaned.iloc[0]["R_early"] == 125
    # Cycle 149 has remaining_cycle 1, should be 1
    engine_1_last_idx = 149  # index for cycle 150
    assert cleaned.iloc[engine_1_last_idx]["R_early"] == 0


def test_test_data_cleaner(sample_test_df):
    test_results, test_df = sample_test_df
    cleaned = cleaner.test_data_cleaner(test_results.copy(), test_df.copy())
    assert "id" not in cleaned.columns
    assert "remaining_cycle" not in cleaned.columns
    assert "sensor2" in cleaned.columns


def test_test_data_cleaner_R_early_no_name_error(sample_test_df):
    """Verify bugfix: ensure no NameError on df_test and R_early computation."""
    test_results, test_df = sample_test_df
    cleaned = cleaner.test_data_cleaner_R_early(test_results.copy(), test_df.copy())
    assert "R_early" in cleaned.columns
    assert "remaining_cycle" not in cleaned.columns
    assert "sensor2" in cleaned.columns
    # Max cycle 30 + RUL 112 = 142. At cycle 1: remaining is 141 -> clipped to 125
    assert cleaned.iloc[0]["R_early"] == 125
