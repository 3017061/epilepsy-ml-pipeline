 # Documentation — Epilepsy ML Pipeline

 This folder contains user-facing documentation that describes the code layout, the core pipeline, and how to run the project.

 ## Quick start

 - Python: 3.8+ recommended

 Setup a virtual environment and install dependencies:

 ```bash
 python -m venv .venv
 # Windows
 .venv\Scripts\activate
 pip install -r requirements.txt
 ```

 Run the smoke test (quick end-to-end check):

 ```bash
 python scripts/smoke_test.py
 # Or on Windows: scripts\run_smoke_test.bat
 ```

 The interactive demo notebook is at the repository root:

 - `epilepsy_ml_pipeline_demo.ipynb`

 ## Project layout (key folders)

 ```
 .
 ├── src/                    # Main code
 │   ├── data/               # Data loading utilities
 │   ├── preprocessing/      # Preprocessing and EEG filters
 │   ├── features/           # Feature extraction & engineering
 │   ├── models/             # Classical and deep model implementations
 │   ├── training/           # Training helpers
 │   ├── evaluation/         # Metrics and evaluation helpers
 │   ├── visualization/      # Plotting utilities
   └── utils/              # Small helpers (seeding, etc.)
 ├── scripts/                # Small runnable scripts (smoke tests)
 ├── notebooks/              # Optional notebooks (may be empty)
 ├── documents/              # This documentation folder
 └── papers/                 # Publication materials
 ```

 ## Primary docs in this folder

 - `PIPELINE_GUIDE.md` — high-level pipeline overview (implementation mapping)
 - `IMPLEMENTATION_SUMMARY.md` — concise mapping of modules to features
 - `BEGINNER_GUIDE.md` — short getting-started instructions
 - `publications.md` — paper and citation metadata

 If you need a runnable experiment runner, use `scripts/smoke_test.py` or open the demo notebook.
