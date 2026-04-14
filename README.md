# Explainable AI Energy Theft Detection

This project demonstrates an explainable AI pipeline for detecting energy theft in smart grids using household consumption patterns.

## Project structure

- `datax/energy_data.csv` - smart meter readings data file (generated if missing).
- `preprocess.py` - data loading, preprocessing, feature engineering, and synthetic dataset generation.
- `model.py` - training, evaluation, and SHAP explainability functions.
- `app.py` - command-line interface for generating data, training, evaluating, and explaining the model.
- `requirements.txt` - Python dependencies.

## Setup

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Generate a synthetic dataset:
  ```bash
  python app.py generate
  ```

- Train a model:
  ```bash
  python app.py train
  ```

- Evaluate the model on the holdout set:
  ```bash
  python app.py evaluate
  ```

- Explain the top suspicious predictions:
  ```bash
  python app.py explain
  ```

## Expected data format

The model expects a CSV with the following columns:
- `meter_id`
- `timestamp`
- `consumption`
- `label` (0 = normal, 1 = theft)

If the data file is missing or empty, the project will generate a synthetic dataset automatically.

## Explainability

This project uses SHAP to provide local explanations for model predictions. That means it can highlight the most important features contributing to each theft risk score.
