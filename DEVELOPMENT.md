# Local Development and Verification Guide

## 1. Upstream Metadata & Licence
- **Original Upstream Repository**: [abiodun-ayodeji/Predictive-Maintenance](https://github.com/abiodun-ayodeji/Predictive-Maintenance)
- **Original Licence**: MIT License (Copyright (c) 2020 Abiodun Ayodeji)
- **Fork Target**: [Kushagra514/turbofan-rul-predictive-maintenance](https://github.com/Kushagra514/turbofan-rul-predictive-maintenance)

## 2. Project Overview
This project predicts the Remaining Useful Life (RUL) of turbofan jet engines based on the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) benchmark dataset (specifically sub-dataset `FD001`). The architecture uses a Deep 1D Convolutional Neural Network (DCNN) with piecewise-linear degradation thresholding ($R_{early} = 125$ cycles) and selected sensor subsets to estimate engine health trajectories.

## 3. Actual Local Folder Structure
```
turbofan-rul/
├── .gitignore                                                    # Git ignore specifications
├── C-MAPSS_data_cleaner.py                                       # Preprocessing helper routines for train/test data
├── DEVELOPMENT.md                                                # Development instructions, audit, and verification status
├── LICENSE                                                       # Original MIT License
├── README.md                                                     # Project documentation with fork provenance
├── Train_data.csv                                                # Cleaned training dataset (FD001 subset)
├── test_engine_21.csv                                            # Sample test sensor readings for engine #21
├── tests/
│   └── test_cleaner.py                                           # Unit and regression tests for data cleaner routines
└── TURBOFAN Engine Remaining Useful Life Prediction with DCNN_main.ipynb # 1D-CNN model training & evaluation notebook
```

## 4. Main Execution Entry Points
- **Preprocessing Helpers**: `C-MAPSS_data_cleaner.py` (provides `train_data_cleaner`, `train_data_cleaner_R_early`, `test_data_cleaner`, and `test_data_cleaner_R_early`).
- **End-to-End Notebook**: `TURBOFAN Engine Remaining Useful Life Prediction with DCNN_main.ipynb` (contains data normalization, sliding window feature extraction, Conv1D model definition, training loop, and RUL prediction evaluation).

## 5. Required Dependencies
- **Core Runtime**: Python 3.10+
- **Data & Scientific**: `pandas`, `numpy`, `scikit-learn`
- **Deep Learning**: `tensorflow` / `keras` (required only for running the DCNN notebook)
- **Testing**: `pytest`

## 6. How to Run Available Tests
Deterministic unit tests for the data cleaning and piecewise-linear RUL clipping logic can be executed via pytest:
```bash
pytest tests/ -v
```

## 7. Current Baseline Verification Status
- **Preprocessing Helpers**: Verified. Unit tests in `tests/test_cleaner.py` passed (4/4 tests passed).
- **DCNN Notebook Training**: Unexecuted during bootstrap to prevent unnecessary compute and CUDA/TensorFlow dependencies in this initial setup phase.

## 8. Known Setup & Code Problems
- **Fixed in this fork**: Upstream `C-MAPSS_data_cleaner.py` contained broken references (`df_train` and `df_test` instead of parameter names `train` and `test`) in both `train_data_cleaner_R_early` and `test_data_cleaner_R_early`. Calling either function raised an immediate `NameError`. Corrected to use local DataFrame parameter references.
- **Notebook File Paths**: Upstream notebook references Google Colab paths (`/content/sample_data/...`).
- **Dependencies Specification**: Upstream lacked a formal `requirements.txt` or `pyproject.toml`.

## 9. Completed vs. Planned Work

### Completed Work
- [x] Cloned and verified upstream Git history and MIT attribution.
- [x] Identified and fixed `NameError` variable referencing bugs in `C-MAPSS_data_cleaner.py`.
- [x] Implemented deterministic unit test suite (`tests/test_cleaner.py`) covering all data cleaning functions.
- [x] Added fork provenance notices and local development documentation.

### Planned Personal Extensions
- [ ] Add a modular CLI training and evaluation script (`train.py` / `evaluate.py`) decoupling the model from Jupyter.
- [ ] Add explicit `requirements.txt` pinned for lightweight CPU/GPU execution.
- [ ] Update notebook dataset loading paths to relative paths.
- [ ] Implement automated GitHub Actions CI workflow to run test suite on PR/push.
