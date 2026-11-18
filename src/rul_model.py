"""
1D Deep Convolutional Neural Network (DCNN) for Remaining Useful Life (RUL) Prediction.

Implements the multi-layer Conv1D architecture from the original research
notebook with configurable sequence length, feature dimensions, and regularization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple, Union

import keras
from keras import layers

logger = logging.getLogger(__name__)


def build_dcnn_model(
    sequence_length: int = 40,
    num_features: int = 14,
    learning_rate: float = 1e-4,
    conv_filters: Sequence[int] = (32, 64, 64, 32),
    kernel_size: int = 8,
    dense_units: int = 100,
    dropout_pool: float = 0.5,
    dropout_dense: float = 0.2,
) -> keras.Model:
    """
    Construct and compile the 1D-CNN regression architecture.

    Architecture summary (following Ayodeji et al. 2021 / notebook implementation):
    1. Input layer: (sequence_length, num_features)
    2. Conv1D layers: 4 stacked convolutional layers with 'same' padding and ReLU.
    3. GlobalMaxPooling1D: aggregates temporal feature maps into a fixed vector.
    4. Dropout: prevents co-adaptation of temporal feature maps.
    5. Fully connected Dense layer: linear transformation with 100 units.
    6. Dropout: regularizes dense layer activations.
    7. Output layer: single scalar regression unit for RUL prediction.

    Loss & Metrics:
    - Huber loss: robust against severe degradation outlier penalties.
    - MAE: interpretable Mean Absolute Error in operational cycles.
    - RMSE: Root Mean Squared Error penalizing large prediction discrepancies.

    Args:
        sequence_length: Number of time steps in input sequence (default 40).
        num_features: Number of sensor channels per time step (default 14).
        learning_rate: RMSprop optimizer learning rate.
        conv_filters: Number of filters per Conv1D layer.
        kernel_size: Temporal convolution window size (default 8).
        dense_units: Number of units in hidden fully-connected layer (default 100).
        dropout_pool: Dropout rate after global pooling (default 0.5).
        dropout_dense: Dropout rate after dense layer (default 0.2).

    Returns:
        Compiled keras.Model instance.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features}")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    model = keras.Sequential(name="turbofan_dcnn_regressor")

    # Explicit Input layer for modern Keras
    model.add(layers.Input(shape=(sequence_length, num_features), name="sensor_sequence_input"))

    # Stacked 1D Convolutions
    for idx, filters in enumerate(conv_filters, start=1):
        model.add(
            layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name=f"conv1d_stage_{idx}",
            )
        )

    # Temporal feature pooling & regularization
    model.add(layers.GlobalMaxPooling1D(name="global_max_pool"))
    if dropout_pool > 0.0:
        model.add(layers.Dropout(dropout_pool, name="dropout_pooling"))

    # Fully connected projection
    model.add(layers.Dense(dense_units, activation="linear", name="dense_features"))
    if dropout_dense > 0.0:
        model.add(layers.Dropout(dropout_dense, name="dropout_dense"))

    # Output regression scalar
    model.add(layers.Dense(1, activation="linear", name="rul_prediction"))

    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    huber_loss = keras.losses.Huber(delta=1.0)
    rmse_metric = keras.metrics.RootMeanSquaredError(name="rmse")

    model.compile(
        optimizer=optimizer,
        loss=huber_loss,
        metrics=["mae", rmse_metric],
    )

    return model


def save_trained_model(
    model: keras.Model,
    filepath: Union[str, Path],
) -> Path:
    """
    Save trained Keras model artifact to disk.

    Args:
        model: Trained keras.Model instance.
        filepath: Destination file path (recommended suffix: .keras).

    Returns:
        Path to saved model file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path


def load_trained_model(
    filepath: Union[str, Path],
) -> keras.Model:
    """
    Load a saved Keras model from disk.

    Args:
        filepath: Path to saved model file.

    Returns:
        Loaded keras.Model instance ready for inference or evaluation.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return keras.models.load_model(path)
