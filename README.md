# VitalSense AI

AI-powered predictive maintenance prototype for data-center equipment.

IMPORTANT: The model is trained on the public AI4I 2020 predictive-maintenance
dataset (synthetic, industrial machines). By the cause of the un-availability of data-center, this is synthetic data-center data.


## What it does
Simulated telemetry -> feature engineering -> failure-risk model -> anomaly detection
-> explanation -> alert -> maintenance recommendation -> live dashboard.

## Tech stack
Python, pandas, scikit-learn, FastAPI, Streamlit, Plotly.

## How to run
1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt
4. python run_all.py
5. Open http://localhost:8501, select SERVER-03 and click "Start degradation".

## Project structure
- notebooks/ : data inspection, EDA and model training
- src/ : features, prediction, anomaly detection, explanation, simulator, fleet
- api/ : FastAPI backend
- dashboard/ : Streamlit dashboard
- models/ : trained models
- scripts/ : smoke tests

## Limitations
Synthetic data, simulated telemetry, risk % is an index (not calibrated).

## Dataset
AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).