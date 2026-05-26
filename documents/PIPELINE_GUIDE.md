# Pipeline Guide — Epilepsy ML Pipeline

This document maps the high-level pipeline to the actual code modules in this repository.

Overview
--------

The pipeline implemented in this repository follows the standard stages:

- Data loading
- Feature extraction & engineering
- Preprocessing (including EEG filters)
- Model definition (classical and deep)
- Training and evaluation
- Visualization of results

Where the code lives
--------------------

- Data loading: `src/data/loader.py` — dataset utilities and splitting helpers.
- Feature extraction: `src/features/extractor.py` and `src/features/engineer.py` — signal/tabular feature functions.
- Preprocessing: `src/preprocessing/pipeline.py` and `src/preprocessing/eeg_filters.py` — scaling, selection, and EEG filters.
- Models: `src/models/classical.py` and `src/models/deep.py` — classical ML factories and PyTorch models.
- Training: `src/training/train.py` — training helpers for sklearn and PyTorch.
- Evaluation: `src/evaluation/metrics.py` — metrics and reporting functions.
- Visualization: `src/visualization/plots.py` — plotting helpers used by notebooks and scripts.
- Utilities: `src/utils/seed.py` — reproducibility helpers.

Running the pipeline
--------------------

- There is no single `experiments/` runner in this repository. Use the smoke test script for a lightweight end-to-end check:

```bash
python scripts/smoke_test.py
```

- For exploratory or research runs, open `epilepsy_ml_pipeline_demo.ipynb` and execute the cells that call into `src/`.

Notes
-----

- Add new experiment runners under `scripts/` or create notebooks that import from `src/`.
- This guide intentionally maps features to existing source modules; consult the modules listed above for API details.
