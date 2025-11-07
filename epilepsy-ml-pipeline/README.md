# Epilepsy ML Pipeline

[![CI](https://github.com/3017061/epilepsy-ml-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/3017061/epilepsy-ml-pipeline/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs%20material-blue)](https://3017061.github.io/epilepsy-ml-pipeline/)

End-to-end baselines for EEG-based seizure prediction inspired by:
**Vaish, S. (2024). The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy. IJFMR.**
Source PDF: https://www.ijfmr.com/papers/2024/3/22870.pdf

## What's inside
- Classical ML: SVM, Random Forest (feature-based: Welch PSD + time stats)
- Deep Learning: 1D CNN, LSTM, CNN+LSTM
- CLI with Typer, unit tests, docs, and GitHub Actions

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
epiml pipeline demo_classic
epiml pipeline demo_deep --model cnn
```

## Citation
If you use this repo, please cite:
> Vaish, S. (2024). The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy. IJFMR, Vol 6, Issue 3. PDF: https://www.ijfmr.com/papers/2024/3/22870.pdf

## Author
**Saniya Vaish**
