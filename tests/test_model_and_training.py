"""
Lightweight unit and integration tests for DCNN model, training pipeline, and evaluation metrics.
All tests use synthetic inputs and execute in seconds without downloading datasets or running full training.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_model import calculate_nasa_score, compute_metrics, main as eval_main
from scripts.train_dcnn import main as train_main
from src.rul_model import build_dcnn_model, load_trained_model, save_trained_model
from src.rul_preprocessing import DEFAULT_FEATURE_COLS, split_by_engine


@pytest.fixture
def synthetic_training_csv(tmp_path: Path) -> Path:
    """Create synthetic CSV with 6 engines to test train/val splitting and 1-epoch training."""
    rows = []
    # 6 engines with 50 cycles each
    for eng_id in range(1, 7):
        for c in range(1, 51):
            row = {"id": eng_id, "cycle": c}
            for col in DEFAULT_FEATURE_COLS:
                row[col] = float(c * 0.1 + (eng_id * 0.5))
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic_train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# 1. Model Architecture & Forward Pass Tests
# ---------------------------------------------------------------------------


def test_model_shapes():
    seq_len = 40
    num_feats = 14
    model = build_dcnn_model(sequence_length=seq_len, num_features=num_feats)

    assert model.input_shape == (None, seq_len, num_feats)
    assert model.output_shape == (None, 1)


def test_model_single_forward_pass():
    model = build_dcnn_model(sequence_length=20, num_features=14)
    dummy_input = np.random.randn(4, 20, 14).astype(np.float32)

    output = model(dummy_input, training=False)
    assert output.shape == (4, 1)
    assert not np.isnan(output.numpy()).any()


def test_model_invalid_parameters():
    with pytest.raises(ValueError, match="sequence_length must be positive"):
        build_dcnn_model(sequence_length=0)

    with pytest.raises(ValueError, match="num_features must be positive"):
        build_dcnn_model(num_features=-2)

    with pytest.raises(ValueError, match="learning_rate must be positive"):
        build_dcnn_model(learning_rate=-0.01)


def test_model_save_and_load(tmp_path: Path):
    model = build_dcnn_model(sequence_length=20, num_features=5)
    save_path = tmp_path / "test_model.keras"

    save_trained_model(model, save_path)
    assert save_path.exists()

    reloaded = load_trained_model(save_path)
    dummy = np.ones((2, 20, 5), dtype=np.float32)
    pred_orig = model.predict(dummy, verbose=0)
    pred_reload = reloaded.predict(dummy, verbose=0)

    np.testing.assert_allclose(pred_orig, pred_reload, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. Evaluation Metric & NASA Score Tests
# ---------------------------------------------------------------------------


def test_perfect_prediction_metrics():
    y_true = np.array([[10.0], [50.0], [100.0]], dtype=np.float32)
    y_pred = y_true.copy()

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2_score"] == 1.0
    assert metrics["nasa_score"] == 0.0


def test_nasa_score_asymmetry():
    """
    NASA score penalizes late predictions (y_pred > y_true, positive diff)
    more heavily than early predictions (y_pred < y_true, negative diff).
    """
    y_true = np.array([[50.0]], dtype=np.float32)

    # Late prediction: model says 60 cycles left (diff = +10) -> component failed earlier!
    late_score = calculate_nasa_score(y_true, np.array([[60.0]], dtype=np.float32))

    # Early prediction: model says 40 cycles left (diff = -10) -> early warning, less catastrophic
    early_score = calculate_nasa_score(y_true, np.array([[40.0]], dtype=np.float32))

    assert late_score > early_score
    # Analytical: exp(10/10) - 1 = e - 1 ~= 1.718; exp(10/13) - 1 = e^0.769 - 1 ~= 1.158
    assert late_score > 1.7
    assert early_score < 1.2


# ---------------------------------------------------------------------------
# 3. Engine-Level Splitting & Leakage Prevention Tests
# ---------------------------------------------------------------------------


def test_split_by_engine_disjoint():
    df = pd.DataFrame({
        "id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        "cycle": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "val": range(10),
    })

    train_df, val_df = split_by_engine(df, val_ratio=0.4, seed=42)

    train_ids = set(train_df["id"].unique())
    val_ids = set(val_df["id"].unique())

    # Strictly disjoint
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids) + len(val_ids) == 5
    assert len(val_ids) == 2


# ---------------------------------------------------------------------------
# 4. End-to-End CLI Smoke Tests (1-Epoch Fast Run)
# ---------------------------------------------------------------------------


def test_training_and_evaluation_pipeline_smoke(tmp_path: Path, synthetic_training_csv: Path):
    train_out = tmp_path / "train_output"
    eval_out = tmp_path / "eval_output"

    # Fast 1-epoch training smoke run on synthetic data
    train_exit = train_main([
        "--data-input", str(synthetic_training_csv),
        "--output-dir", str(train_out),
        "--sequence-length", "20",
        "--val-ratio", "0.33",
        "--epochs", "1",
        "--batch-size", "16",
    ])
    assert train_exit == 0
    assert (train_out / "best_model.keras").exists()
    assert (train_out / "history.json").exists()
    assert (train_out / "training_summary.json").exists()

    with open(train_out / "training_summary.json") as f:
        summary = json.load(f)
    assert summary["epochs_trained"] == 1
    assert summary["val_engines_count"] == 2

    # Run evaluation script on the trained model
    eval_exit = eval_main([
        "--model-path", str(train_out / "best_model.keras"),
        "--data-input", str(synthetic_training_csv),
        "--history-path", str(train_out / "history.json"),
        "--output-dir", str(eval_out),
        "--sequence-length", "20",
        "--val-ratio", "0.33",
    ])
    assert eval_exit == 0
    assert (eval_out / "evaluation_metrics.json").exists()
    assert (eval_out / "actual_vs_predicted.png").exists()
    assert (eval_out / "residuals_distribution.png").exists()
    assert (eval_out / "loss_and_metrics_curves.png").exists()

    with open(eval_out / "evaluation_metrics.json") as f:
        metrics = json.load(f)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "nasa_score" in metrics


def test_train_missing_file_returns_error(tmp_path: Path):
    exit_code = train_main([
        "--data-input", str(tmp_path / "non_existent.csv"),
    ])
    assert exit_code == 1
