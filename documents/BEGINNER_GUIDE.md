# Beginner Guide — Getting started

This short guide shows the minimum steps to get the project running locally.

1) Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the smoke test (quick end-to-end verification)

```bash
python scripts/smoke_test.py
```

4) Open the demo notebook for interactive exploration

Open `epilepsy_ml_pipeline_demo.ipynb` from the repository root in Jupyter.

Where to look in the code
-------------------------

- Data: `src/data/loader.py`
- Preprocessing: `src/preprocessing/pipeline.py`, `src/preprocessing/eeg_filters.py`
- Features: `src/features/engineer.py`, `src/features/extractor.py`
- Models: `src/models/classical.py`, `src/models/deep.py`
- Training: `src/training/train.py`
- Evaluation: `src/evaluation/metrics.py`

If you want help running a specific example or creating an experiment runner, tell me which dataset or model you'd like to try and I can create a small script/notebook for it.
