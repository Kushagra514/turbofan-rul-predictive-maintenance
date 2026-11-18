# Local Development and Verification Guide

## 1. Upstream Metadata & Licence
- **Original Upstream Repository**: [abiodun-ayodeji/Predictive-Maintenance](https://github.com/abiodun-ayodeji/Predictive-Maintenance)
- **Original Licence**: MIT License (Copyright (c) 2020 Abiodun Ayodeji)
- **Fork Target**: [Kushagra514/turbofan-rul-predictive-maintenance](https://github.com/Kushagra514/turbofan-rul-predictive-maintenance)

## 2. Project Overview
This project predicts the Remaining Useful Life (RUL) of turbofan jet engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) benchmark dataset (`FD001` subset). The system implements an end-to-end, reproducible machine learning pipeline consisting of:
1. **Data Preprocessing**: Piecewise-linear degradation clipping ($R_{early} = 125$ cycles), leak-free sensor normalization, and temporal sliding-window sequence extraction.
2. **1D-CNN Regression Architecture**: Multi-stage 1D convolutional feature extractor with global temporal pooling.
3. **Reproducible Training Pipeline**: Engine-level train/validation splitting, early stopping, and model checkpointing.
4. **Model Evaluation & Diagnostics**: MAE, RMSE, $R^2$, and the official NASA C-MAPSS asymmetric scoring function, accompanied by diagnostic visualizations.

## 3. Actual Local Folder Structure
```
turbofan-rul/
├── .gitignore                                                    # Excludes virtual environments, caches, data/, and artifacts/
├── C-MAPSS_data_cleaner.py                                       # Original C-MAPSS feature cleaning routines
├── DEVELOPMENT.md                                                # Development instructions, audit, and verification status
├── LICENSE                                                       # Original MIT License
├── README.md                                                     # Project documentation with fork provenance and CLI usage
├── Train_data.csv                                                # Cleaned training dataset (100 engines, 20,631 cycles)
├── test_engine_21.csv                                            # Cleaned test dataset (engine #21, 195 cycles)
├── scripts/
│   ├── prepare_data.py                                           # CLI for local data preparation and sequence generation
│   ├── train_dcnn.py                                             # CLI for reproducible DCNN training with engine-level splitting
│   └── evaluate_model.py                                         # CLI for model evaluation, metrics, and diagnostic plots
├── src/
│   ├── __init__.py                                               # Package exports
│   ├── rul_model.py                                              # 1D-CNN architecture definition and model persistence
│   └── rul_preprocessing.py                                      # Modular data loading, scaling, splitting, and sequence generation
├── tests/
│   ├── conftest.py                                               # Pytest path and environment configuration
│   ├── test_cleaner.py                                           # Regression tests for C-MAPSS_data_cleaner.py (4 tests)
│   ├── test_model_and_training.py                                # DCNN model, metrics, and training smoke tests (9 tests)
│   └── test_preprocessing.py                                     # Unit & integration tests for preprocessing pipeline (14 tests)
└── TURBOFAN Engine Remaining Useful Life Prediction with DCNN_main.ipynb # Original research notebook
```

## 4. CNN Architecture & Specifications

The 1D-CNN architecture (`src/rul_model.py`) models temporal sensor dynamics without requiring recurrent connections:

```
Input: (batch_size, sequence_length=40, num_features=14)
   │
   ├─► Conv1D(32 filters, kernel_size=8, padding='same', activation='relu')   -> (None, 40, 32)
   ├─► Conv1D(64 filters, kernel_size=8, padding='same', activation='relu')   -> (None, 40, 64)
   ├─► Conv1D(64 filters, kernel_size=8, padding='same', activation='relu')   -> (None, 40, 64)
   ├─► Conv1D(32 filters, kernel_size=8, padding='same', activation='relu')   -> (None, 40, 32)
   │
   ├─► GlobalMaxPooling1D()                                                  -> (None, 32)
   ├─► Dropout(rate=0.5)
   │
   ├─► Dense(100 units, activation='linear')                                  -> (None, 100)
   ├─► Dropout(rate=0.2)
   │
   └─► Dense(1 unit, activation='linear') [Output RUL Prediction]            -> (None, 1)
```

- **Total Parameters**: 72,713 trainable parameters (~284 KB in memory).
- **Loss Function**: Huber loss ($\delta = 1.0$), which behaves quadratically for small residuals and linearly for large residuals, mitigating sensitivity to degradation outliers.
- **Optimizer**: RMSprop (learning rate $\eta = 10^{-4}$).

## 5. Data Pipeline & Leakage Prevention Strategy

### A. Engine-Level Splitting (Zero Window Leakage)
In time-series sequence modeling, randomly splitting sliding windows produces artificial data leakage because window $i$ (cycles $t \dots t+W-1$) and window $i+1$ (cycles $t+1 \dots t+W$) share $W-1$ identical timesteps.

Our pipeline enforces **engine-level disjoint splitting**:
- Unique engine IDs are partitioned into training engines (80%) and validation engines (20%).
- All operational cycles for any given engine reside strictly within one partition.

### B. Preprocessing Isolation (Zero Statistical Leakage)
- Feature normalization statistics (Min/Max values) are computed strictly from the training engines.
- The validation engines are transformed using the fitted training scaler without computing test statistics.

### C. Temporal Sequence Alignment
- A sliding window of length $W = 40$ spans cycles $[t - W + 1 \dots t]$.
- The aligned target label $y_t = R_{early}(t+1)$ represents the remaining useful life at the step immediately following the window.

## 6. How to Run Training and Evaluation

### Training the DCNN Model
```bash
python scripts/train_dcnn.py \
  --data-input Train_data.csv \
  --output-dir artifacts/training \
  --val-ratio 0.2 \
  --sequence-length 40 \
  --rul-clip 125 \
  --epochs 15 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --patience 5 \
  --seed 42
```

Outputs written to `artifacts/training/`:
- `best_model.keras`: Checkpointed best model weights based on minimum validation loss.
- `history.json`: Epoch-by-epoch loss, MAE, and RMSE metrics.
- `training_summary.json`: Detailed execution metadata, engine IDs, sample counts, and final scores.
- `scaler_params.json`: Feature normalization bounds for standalone inference.

### Evaluating the Model
```bash
python scripts/evaluate_model.py \
  --model-path artifacts/training/best_model.keras \
  --data-input Train_data.csv \
  --history-path artifacts/training/history.json \
  --output-dir artifacts/evaluation \
  --val-ratio 0.2 \
  --seed 42
```

Outputs written to `artifacts/evaluation/`:
- `evaluation_metrics.json`: Quantitative report (MAE, RMSE, $R^2$, NASA Score).
- `actual_vs_predicted.png`: Time-series overlay of predicted versus ground truth RUL.
- `residuals_distribution.png`: Histogram and scatter plot of prediction errors.
- `loss_and_metrics_curves.png`: Training vs validation loss and RMSE progression curves.

## 7. Actual Measured Baseline Performance

The model was trained on the NASA C-MAPSS dataset on Ubuntu Linux (CPU execution):
- **Training Engines**: 80 engines (13,361 sequences)
- **Validation Engines**: 20 held-out engines (3,270 sequences)
- **Training Epochs**: 15 epochs (best weights restored from epoch 14)

### Held-Out Validation Set Metrics (3,270 samples):
| Metric | Measured Value | Interpretation |
|---|---|---|
| **Huber Loss** | **12.7217** | Robust regression objective |
| **MAE** | **13.21 cycles** | Mean absolute prediction error across all operational cycles |
| **RMSE** | **17.52 cycles** | Standard deviation of prediction errors |
| **$R^2$ Score** | **0.8221** | Model explains 82.2% of the variance in engine RUL |
| **NASA Score** | **21,891.1** | Official asymmetric PHM challenge penalty score |
| **Mean Error** | **-2.01 cycles** | Slight tendency to predict RUL 2 cycles early (safer for maintenance) |

## 8. Unit Test Suite (27 Passed Tests)

Run the complete test suite:
```bash
pytest tests/ -v
```

Test breakdown:
- **`tests/test_cleaner.py` (4 tests)**: Validates fix for upstream `NameError` bugs and original cleaner functions.
- **`tests/test_preprocessing.py` (14 tests)**: Tests CSV loading, schema validation, piecewise RUL calculation, leak-free scaling, sliding-window generation, and CLI commands.
- **`tests/test_model_and_training.py` (9 tests)**: Tests model input/output shapes, single forward pass, model saving/loading, NASA score asymmetry, engine-level splitting, and 1-epoch CLI smoke tests.

## 9. Development Environment & Verified Dependencies

Development environment on Ubuntu Linux (Lenovo LOQ):
- **Python**: 3.14.6
- **TensorFlow (CPU)**: 2.22.0-rc1
- **Keras**: 3.16.0
- **scikit-learn**: 1.9.0
- **pandas**: 3.0.3
- **numpy**: 2.4.6
- **matplotlib**: 3.11.0
- **pytest**: 9.0.3

*Zero heavy GPU or CUDA dependencies were required. CPU execution with AVX instructions trains 15 epochs in ~25 seconds.*

## 10. Completed vs. Planned Work

### Completed Work
- [x] Pre-training audit of sequence alignment, engine separation, and data leakage.
- [x] Identified that `test_engine_21.csv` is an extract of Engine 21 from `Train_data.csv`, and established strict engine-level validation splitting to prevent evaluating on training data.
- [x] Extracted modular 1D-CNN architecture (`src/rul_model.py`).
- [x] Built configurable training pipeline (`scripts/train_dcnn.py`) with early stopping, checkpointing, and engine-level splitting.
- [x] Built evaluation pipeline (`scripts/evaluate_model.py`) with MAE, RMSE, $R^2$, NASA score, and diagnostic plots.
- [x] Verified full test suite with 27 automated tests passing in ~6.5 seconds.
- [x] Successfully trained model on 80 engines and achieved 13.21 MAE / 17.52 RMSE on 20 held-out engines.

### Planned Personal Extensions
- [ ] Add inference CLI (`scripts/predict_rul.py`) for streaming sensor batches.
- [ ] Experiment with dilated convolutions or causal temporal convolutions (TCN).
- [ ] Implement automated GitHub Actions CI workflow to run test suite on PR/push.
