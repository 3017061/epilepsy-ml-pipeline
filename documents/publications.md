# Publication Metadata

Source Code repo.

**Repository cleanup (2026-05-26):** Compiled caches (`.pyc` / `__pycache__`) were removed and a `.gitignore` was added to prevent these files from being committed. No source code files were removed.

## Manuscript Information

### Title
**The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy**

This research explores machine learning approaches for epilepsy prediction and diagnosis using biomedical data analysis techniques.

### Authors
1. **Primary Author (Saniya Vaish)**

Author contributions: Concept design, methodology, data analysis, implementation, and manuscript preparation.

### Abstract
Epilepsy affects over 50 million people worldwide, with diagnosis often relying on EEG signals and clinical expertise. This research investigates machine learning approaches for automated epilepsy prediction and neurological disorder detection. We implement and compare multiple classical machine learning models (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes) and deep neural networks on biomedical datasets. Our methodology includes comprehensive data preprocessing, feature engineering with polynomial and ratio-based features, and rigorous cross-validation evaluation. Results demonstrate that ensemble methods (Random Forest, Gradient Boosting) achieve superior performance for epilepsy classification. The implementation provides reproducible workflows suitable for clinical decision support systems. This work contributes to the field of computational neurology by providing scalable, interpretable ML pipelines for neurological disorder prediction.

### Publication Details
- **Status**: Published
- **Venue**: International Journal for Multidisciplinary Research (IJFMR)
- **Volume/Issue**: 6(3), May-June 2024
- **ISSN**: 2582-2160
- **DOI**: 10.36948/ijfmr.2024.v06i03.22870
- **URL**: https://www.ijfmr.com

### Research Keywords
- Epilepsy Detection
- EEG Signal Processing
- Machine Learning Classification
- Neurological Disorder Prediction
- Computational Neurology
- Deep Learning for Healthcare
- Feature Engineering
- Biomedical Signal Analysis

## Repository Structure for Publication

This repository implements the computational methods described in the manuscript:

### Datasets
- **Primary Dataset**: Breast Cancer Wisconsin Dataset (UCI ML Repository)
- **Supplementary Data**: Wine Classification Dataset (UCI ML Repository)
- License and citation information included in `src/data/loader.py`

### Methods Implemented

#### 1. Data Preprocessing
- Missing value imputation (median strategy)
- Feature scaling (standardization)
- Outlier detection and handling
- **File**: `src/preprocessing/pipeline.py`

#### 2. Feature Engineering
- Ratio-based feature construction
- Polynomial feature generation (degree 2)
- Statistical feature selection (f-statistic)
- **File**: `src/features/engineer.py`

#### 3. Classification Models
- **Classical Models**:
  - Logistic Regression (L2 regularization)
  - Decision Trees (CART)
  - Random Forest (200 estimators)
  - Gradient Boosting
  - Support Vector Machines (RBF kernel)
  - K-Nearest Neighbors
  - Naive Bayes
- **Deep Learning**: PyTorch MLP with dropout regularization
- **Files**: `src/models/classical.py`, `src/models/deep.py`

#### 4. Model Evaluation
- Cross-validation (stratified k-fold)
- Hyperparameter optimization (GridSearchCV)
- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **File**: `src/evaluation/metrics.py`

#### 5. Visualization
- Exploratory Data Analysis (EDA)
- Feature correlation heatmaps
- PCA embeddings
- Confusion matrices
- Feature importance plots
- **File**: `src/visualization/plots.py`

### Reproducibility

All results can be reproduced using:
```bash
python experiments/run_showcase.py
jupyter notebook notebooks/ml_showcase.ipynb
```

Fixed random seeds ensure reproducibility across runs (see `src/utils/seed.py`).

### Code Availability

This repository is publicly available under the MIT License at:
```
https://github.com/3017061/epilepsy-ml-pipeline
```

## Citation

If you use this code or methodology in your research, please cite:

### BibTeX
```bibtex
@article{vaish2024epilepsy,
  author = {Vaish, Saniya},
  title = {The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy},
  journal = {International Journal for Multidisciplinary Research (IJFMR)},
  volume = {6},
  number = {3},
  pages = {May-June},
  year = {2024},
  issn = {2582-2160},
  doi = {10.36948/ijfmr.2024.v06i03.22870},
  url = {https://www.ijfmr.com}
}

@software{vaish2026showcaserepo,
  author = {Vaish, Saniya},
  title = {Biomedical ML Research Showcase: Epilepsy Prediction Pipeline},
  year = {2026},
  url = {https://github.com/3017061/epilepsy-ml-pipeline},
  note = {Open-source implementation accompanying IJFMR publication}
}
```

### APA
Vaish, S. (2024). The use of machine learning in predicting neurological disorders for epilepsy. *International Journal for Multidisciplinary Research (IJFMR)*, 6(3), May-June. https://doi.org/10.36948/ijfmr.2024.v06i03.22870

Vaish, S. (2026). Biomedical ML research showcase: Epilepsy prediction pipeline [Computer software]. Retrieved from https://github.com/3017061/epilepsy-ml-pipeline


## Contact

**Corresponding Author**: 
- **Name**: Saniya Vaish
- **Email**: 3017061@gmail.com
- **Institution**: John P. Stevens High School, Edison, NJ, USA

---
