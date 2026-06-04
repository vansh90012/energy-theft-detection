# Explainable AI Energy Theft Detection

This project demonstrates an explainable AI pipeline for detecting energy theft in smart grids using household consumption patterns.

## Project structure

- `datax/energy_data.csv` - smart meter readings data file (generated if missing).
- `preprocess.py` - data loading, preprocessing, feature engineering, and synthetic dataset generation.
- `model.py` - training, evaluation, and SHAP explainability functions.
- `app.py` - command-line interface for generating data, training, evaluating, and explaining the model.
- `requirements.txt` - Python dependencies.
- `models/theft_detector.pkl` - saved trained model file created by `app.py train`.

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

- Train a model with the default dataset (saved to `models/theft_detector.pkl`):
  ```bash
  python app.py train
  ```

- Evaluate the model on the default dataset:
  ```bash
  python app.py evaluate
  ```

- Force retrain and overwrite the saved model:
  ```bash
  python app.py train --force-retrain
  ```

- Run the Streamlit dashboard for visual analysis and explanations:
  ```bash
  python -m streamlit run dashboard.py
  ```

## Kaggle dataset support

This project supports real Kaggle datasets and can convert wide-format Kaggle files into the long format used by the model.

Example download using the Kaggle CLI:

```bash
pip install kaggle
mkdir -p datax
kaggle datasets download -d walagooose/sgcc-after-processing -p datax
unzip datax/sgcc-after-processing.zip -d datax
```

Then run training with the SGCC files:

```bash
python app.py train --data-path datax/after_preprocess_data.csv --label-path datax/label.csv
python app.py evaluate --data-path datax/after_preprocess_data.csv --label-path datax/label.csv
```

For a single-file Kaggle dataset, just pass the path directly:

```bash
python app.py train --data-path datax/Electricity_Theft_Data.csv
```

## Expected data format

The default model input is a CSV with one row per meter reading and the following columns:
- `meter_id`
- `timestamp`
- `consumption`
- `theft` or `label` (0 = normal, 1 = theft)

The loader also supports common Kaggle styles:
- separate `label.csv` files
- wide-format datasets with one row per customer and date columns

If the data file is missing, the project will generate a synthetic dataset automatically.

## Explainability

This project uses SHAP to provide local explanations for model predictions. That means it can highlight the most important features contributing to each theft risk score.

## Live Hosting

This repository is ready to deploy as a Streamlit app. The public URL after deployment on Streamlit Community Cloud will be:

`https://share.streamlit.io/vansh90012/energy-theft-detection/main/dashboard.py`

To publish it:

1. Push this repository to GitHub.
2. Go to https://share.streamlit.io and log in with your GitHub account.
3. Create a new app using the repository `vansh90012/energy-theft-detection`, branch `main`, and file `dashboard.py`.
4. Streamlit will install `requirements.txt` and launch the app.

Once deployed, the app will be live at the URL above.
