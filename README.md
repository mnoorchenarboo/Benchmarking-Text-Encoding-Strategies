# Benchmarking Text Encoding Strategies for Surgical Duration Prediction

This repository benchmarks multiple text encoding strategies for predicting surgical case duration.
It compares structured-only baselines, classical text features, and BERT-based embeddings across multiple regression models.

## What this project does

- Builds a cleaned tabular dataset from raw surgical case data.
- Generates text representations using:
	- Structured only baseline
	- Label encoding
	- Count vectorization
	- TF-IDF
	- ClinicalBERT embeddings
	- Sentence-BERT embeddings
- Trains and evaluates several models:
	- Linear Regression
	- Ridge
	- Lasso
	- Random Forest
	- XGBoost
	- MLP (TensorFlow/Keras)
- Reports metrics including MAE, SMAPE, R2, RMSE, training time, and inference time.
- Supports both standard K-Fold CV and alternative Hospital/Temporal CV strategies.

## Repository structure

```text
.
|-- pipeline.py                # Main 4-stage pipeline (K-Fold workflow)
|-- pipeline_cv.py             # Alternative CV pipeline (hospital + temporal)
|-- dashboard.py               # Flask + Plotly interactive dashboard
|-- OverLeaf.ipynb             # Figure generation notebook for publication-ready plots
|-- data/
|   |-- casetime.csv           # Input dataset
|   |-- bert_cache/            # Cached BERT embeddings (.npy)
|-- results/                   # Outputs for pipeline.py
|-- results_hospital/          # Outputs for hospital CV
|-- results_temporal/          # Outputs for temporal CV
|-- overleaf/                  # Exported PDF figures/logs from notebook
|-- sync.ps1                   # Optional git sync helper script
```

## Pipeline overview

### `pipeline.py` (main workflow)

The script is interactive and organized into four stages:

1. Stage 01 - Pre-processing
	 - Reads `data/casetime.csv`
	 - Writes cleaned data and fold indices into `data/surgical_data.db`
2. Stage 02 - BERT cache
	 - Creates reusable embedding caches under `data/bert_cache/`
3. Stage 03 - Fold encoding
	 - Performs fold-safe imputation/encoding
	 - Applies PCA to BERT embeddings per fold (train-only)
	 - Saves encoded matrices in `data/fold_encoded.db`
4. Stage 04 - Modeling
	 - Tunes, trains, and evaluates selected models
	 - Saves metrics/artifacts in `results/result.db` and `results/*.log`, `results/*.pdf`

Notes:
- Stages 01-03 auto-skip when already complete.
- Stage 04 always prompts for model selection.

### `pipeline_cv.py` (alternative CV workflow)

Adds two CV strategies on top of the cleaned dataset:

- Hospital CV: leave-one-location-out style split
- Temporal CV: expanding time-series split

It writes separate outputs:

- Encoded DBs: `data/fold_encoded_hospital.db`, `data/fold_encoded_temporal.db`
- Results DBs: `results_hospital/result.db`, `results_temporal/result.db`
- Logs/plots in `results_hospital/` and `results_temporal/`

Prerequisites for `pipeline_cv.py`:

- `pipeline.py` Stage 01 completed
- BERT cache available (from `pipeline.py` Stage 02)

## Requirements

Python 3.10+ is recommended.

Core libraries used by this project include:

- numpy
- pandas
- scipy
- scikit-learn
- optuna
- xgboost
- tensorflow
- torch
- transformers
- sentence-transformers
- matplotlib
- flask
- plotly (loaded via CDN in dashboard frontend)

Example install command:

```bash
pip install numpy pandas scipy scikit-learn optuna xgboost tensorflow torch transformers sentence-transformers matplotlib flask
```

## How to run

### 1) Main benchmark pipeline

```bash
python pipeline.py
```

You will be prompted to run selected stages (or all stages by default).

### 2) Alternative CV experiments

```bash
python pipeline_cv.py
```

You will be prompted for stage selection, CV strategy, and models.

### 3) Dashboard

```bash
python dashboard.py
```

Then open:

- `http://localhost:5050`

The dashboard reads from `results/result.db`.

### 4) Publication figures

Open and run `OverLeaf.ipynb` to export sensitivity plots and legend PDFs into `overleaf/`.

## Outputs and artifacts

Main outputs you can expect:

- Databases
	- `data/surgical_data.db`
	- `data/fold_encoded.db`
	- `results/result.db`
	- `results_hospital/result.db`
	- `results_temporal/result.db`
- Logs
	- Stage logs: `results/*.log`, `results_hospital/*.log`, `results_temporal/*.log`
- Figures
	- Model comparison PDFs in results folders
	- Sensitivity analysis PDFs in `overleaf/`

## Reproducibility notes

- Most random operations are configured with fixed seeds in pipeline config blocks.
- Stage 03 operations are fold-aware to reduce leakage risk.
- BERT features are cached so repeated runs are faster and consistent.

## Data and privacy

Before publishing this repository publicly:

- Confirm that `data/casetime.csv` is de-identified and approved for sharing.
- Verify no sensitive artifacts are present in generated databases/logs.
- Consider adding a `.gitignore` policy for large outputs and local DB files if needed.

## Optional helper script

`sync.ps1` is a convenience script for local git workflow automation (index generation, add/commit/pull/push). Review it before use, especially in collaborative branches.
