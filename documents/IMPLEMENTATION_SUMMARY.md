# Implementation Summary — Epilepsy ML Pipeline

This file gives a concise mapping between implemented features and the source modules. It is intended as a quick reference for developers.

Core components
---------------

- Data loading: `src/data/loader.py` — functions to load public datasets and split data for experiments.
- Feature extraction: `src/features/extractor.py`, `src/features/engineer.py` — signal feature extraction and tabular feature engineering utilities.
- Preprocessing: `src/preprocessing/pipeline.py`, `src/preprocessing/eeg_filters.py` — pipelines for scaling, selection, and EEG-specific filtering.
- Models: `src/models/classical.py`, `src/models/deep.py` — factory functions and network definitions for classical and deep learning models.
- Training: `src/training/train.py` — training wrappers and helper functions for sklearn and PyTorch workflows.
- Evaluation: `src/evaluation/metrics.py` — classification and medical-relevant metrics (sensitivity, specificity, ROC, etc.).
- Visualization: `src/visualization/plots.py` — plotting utilities used by notebooks and scripts.
- Utilities: `src/utils/seed.py` — reproducible random seeds and small helpers.

How to extend
-------------

- Add new datasets under `src/data/` and expose them via `load_public_dataset()`.
- Implement new models in `src/models/` and add a factory wrapper to keep usage consistent.
- For experiments, create scripts in `scripts/` or notebooks that import `src` modules.

Notes
-----

This summary intentionally avoids claiming non-existent top-level runners. Use `scripts/smoke_test.py` for a standard quick run, and the notebook for interactive experiments.
