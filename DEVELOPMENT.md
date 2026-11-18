# Local Development and Verification Guide

## 1. Upstream Metadata & Licence
- **Original Upstream Repository**: [abiodun-ayodeji/Predictive-Maintenance](https://github.com/abiodun-ayodeji/Predictive-Maintenance)
- **Original Licence**: MIT License (Copyright (c) 2020 Abiodun Ayodeji)
- **Fork Target**: [Kushagra514/turbofan-rul-predictive-maintenance](https://github.com/Kushagra514/turbofan-rul-predictive-maintenance)

## 2. Project Overview
This project predicts the Remaining Useful Life (RUL) of turbofan jet engines using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) benchmark dataset (`FD001` subset). The model architecture is a Deep 1D Convolutional Neural Network (DCNN) that accepts sliding-window sensor sequences and predicts a piecewise-linear rectified RUL ($R_{early} = 125$ cycles).

## 3. Actual Local Folder Structure
```
turbofan-rul/
├── .gitignore                                                    # Excludes virtual environments, caches, and data/
├── C-MAPSS_data_cleaner.py                                       # Original C-MAPSS feature cleaning routines
├── DEVELOPMENT.md                                                # Development instructions, audit, and verification status
├── LICENSE                                                       # Original MIT License
├── README.md                                                     # Project documentation with fork provenance and CLI usage
├── Train_data.csv                                                # Cleaned training dataset (100 engines, 20,631 cycles)
├── test_engine_21.csv                                            # Cleaned test dataset (engine #21, 195 cycles)
├── scripts/
│   └── prepare_data.py                                           # Command-line interface for local data preparation
├── src/
│   ├── __init__.py                                               # Package exports
│   └── rul_preprocessing.py                                      # Modular data loading, scaling, and sequence generation
├── tests/
│   ├── conftest.py                                               # Pytest path and environment configuration
│   ├── test_cleaner.py                                           # Regression tests for C-MAPSS_data_cleaner.py (4 tests)
│   └── test_preprocessing.py                                     # Unit & integration tests for modular pipeline (14 tests)
└── TURBOFAN Engine Remaining Useful Life Prediction with DCNN_main.ipynb # 1D-CNN model training & evaluation notebook
```

## 4. Preprocessing Architecture & Pipeline Stages

```
   Raw / Cleaned CSVs (Train & Test)
                 │
                 ▼
  [ Stage 1: Load & Validate ]
   - Validate file existence and non-emptiness
   - Check required schema (id, cycle, sensor columns)
                 │
                 ▼
  [ Stage 2: Piecewise-Linear RUL Computation ]
   - remaining_cycle = max(cycle) - cycle  (per engine id)
   - R_early = min(remaining_cycle, rul_clip=125)
                 │
                 ▼
  [ Stage 3: Leak-Free Normalization ]
   - Fit MinMaxScaler strictly on Train feature columns
   - Transform Train features and Test features independently
                 │
                 ▼
  [ Stage 4: Temporal Sliding-Window Extraction ]
   - Sliding window of length W (default: 40 cycles)
   - Input tensor X: shape (N, 40, 14)
   - Output target y: shape (N, 1) aligned with window end
                 │
                 ▼
  [ Stage 5: Artifact Persistence ]
   - train_normalized.csv, test_normalized.csv
   - train_sequences.npz, test_sequences.npz
   - metadata.json (dataset statistics & tensor shapes)
```

## 5. Input Data Format & Expected Columns

The preprocessing module expects tabular data with the following schema:
- **Identifier**: `id` (integer engine ID).
- **Time/Sequence**: `cycle` (operational cycle number, starting at 1).
- **Sensor Features (14 channels)**:
  `sensor2`, `sensor3`, `sensor4`, `sensor7`, `sensor8`, `sensor9`, `sensor11`, `sensor12`, `sensor13`, `sensor14`, `sensor15`, `sensor17`, `sensor20`, `sensor21`.
- **Target Label (optional on input)**: `R_early` (piecewise-linear RUL). If absent, it is calculated automatically from engine cycles.

## 6. How the Command-Line Tool Works

The preprocessing script `scripts/prepare_data.py` provides a standalone CLI with argument parsing, validation, and progress reporting:

### Example CLI Execution
```bash
python scripts/prepare_data.py \
  --train-input Train_data.csv \
  --test-input test_engine_21.csv \
  --output-dir data/processed \
  --sequence-length 40 \
  --rul-clip 125
```

### CLI Arguments
| Flag | Type | Default | Description |
|---|---|---|---|
| `--train-input` | string | *required* | Path to training CSV file |
| `--test-input` | string | `None` | Path to optional test CSV file |
| `--output-dir` | string | `./data/processed` | Destination directory for processed artifacts |
| `--sequence-length`| integer| `40` | Temporal window size in operational cycles |
| `--rul-clip` | integer| `125` | Early degradation threshold cycle |

### Output Artifacts
- `train_normalized.csv`: Scaled tabular training dataset.
- `test_normalized.csv`: Scaled tabular test dataset (when test input is provided).
- `train_sequences.npz`: Compressed NumPy archive containing:
  - `X`: shape `(16631, 40, 14)`, `float32`.
  - `y`: shape `(16631, 1)`, `float32`.
- `test_sequences.npz`: Compressed NumPy archive containing:
  - `X`: shape `(155, 40, 14)`, `float32`.
  - `y`: shape `(155, 1)`, `float32`.
- `metadata.json`: JSON record of dataset shapes, engine counts, and feature names.

## 7. How to Run Unit Tests

Tests use small synthetic DataFrames and temporary directories, running entirely offline without depending on external dataset downloads:

```bash
pytest tests/ -v
```

### Current Test Suite (18 passed in ~1.0s)
- **`tests/test_cleaner.py` (4 tests)**:
  - `test_train_data_cleaner`: Verifies standard RUL computation and column dropping.
  - `test_train_data_cleaner_R_early_no_name_error`: Verifies fix for upstream `df_train` NameError.
  - `test_test_data_cleaner`: Verifies test ground-truth RUL merge.
  - `test_test_data_cleaner_R_early_no_name_error`: Verifies fix for upstream `df_test` NameError.
- **`tests/test_preprocessing.py` (14 tests)**:
  - `test_load_csv_data_valid`: Valid CSV ingestion and shape validation.
  - `test_load_csv_data_missing_file`: FileNotFoundError on non-existent path.
  - `test_load_csv_data_empty_file`: ValueError on empty files.
  - `test_load_csv_data_missing_columns`: ValueError when required columns are missing.
  - `test_compute_piecewise_rul`: Correct thresholding at 125 cycles and linear descent.
  - `test_compute_piecewise_rul_missing_columns`: ValueError on missing metadata columns.
  - `test_scaler_no_data_leakage`: Confirms scaler is fit solely on train; preserves metadata.
  - `test_generate_sequences_shapes_and_alignment`: 3D tensor shape verification and label alignment.
  - `test_generate_sequences_invalid_sequence_length`: Validation for invalid window parameters.
  - `test_process_pipeline_deterministic`: Validates identical outputs across repeated runs.
  - `test_save_processed_outputs_creates_directory_and_artifacts`: Auto-creation of directory and persistence.
  - `test_cli_success`: CLI smoke test on synthetic data returning exit code 0.
  - `test_cli_missing_file_returns_error`: CLI returns non-zero exit code (1) on missing input.
  - `test_cli_invalid_arguments`: CLI returns non-zero exit code (1) on invalid parameters.

## 8. Development Environment & Verified Dependencies

Development and verification environment on Ubuntu Linux (Lenovo LOQ, RTX 3050 6GB):
- **Python**: 3.14.6 (Anaconda base environment)
- **pandas**: 3.0.3
- **numpy**: 2.4.6
- **scikit-learn**: 1.9.0
- **pytest**: 9.0.3

*Note: Preprocessing requires zero GPU or CUDA packages. Full model training with TensorFlow/Keras remains a separate downstream milestone.*

## 9. Completed vs. Planned Work

### Completed Work
- [x] Fixed upstream `NameError` bugs in `C-MAPSS_data_cleaner.py`.
- [x] Extracted modular preprocessing pipeline (`src/rul_preprocessing.py`).
- [x] Built command-line preprocessing tool (`scripts/prepare_data.py`).
- [x] Prevented test-set data leakage via strict train-only scaler fitting.
- [x] Implemented comprehensive synthetic unit test suite (18 tests passing).
- [x] Updated `.gitignore` to prevent accidental staging of generated data artifacts (`data/`, `*.npz`).
- [x] Documented local CLI workflow and dataset schemas.

### Planned Personal Extensions
- [ ] Implement modular model training script (`src/train_dcnn.py`) to run 1D-CNN training directly from generated `.npz` sequence arrays.
- [ ] Add inference / prediction CLI (`scripts/predict_rul.py`) evaluating trained models on new engine trajectories.
- [ ] Add automated evaluation metrics (RMSE, NASA scoring function $s = \sum (\exp(d/13) - 1)$).
- [ ] Create GitHub Actions CI workflow to run test suite on push and PR.
