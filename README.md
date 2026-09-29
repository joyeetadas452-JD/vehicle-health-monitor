# AI/ML-Based Predictive Maintenance & Vehicle Health Monitoring System
Module 3 (Connectivity) – Project 1

## Pipeline
Vehicle sensors → Preprocessing (scaling) → ML model → Failure probability → Health score → Maintenance alert

## Inputs (9)
Engine temp, RPM, oil pressure, vibration, battery voltage, coolant temp, fuel consumption, speed, operating hours.

## Models compared
Logistic Regression, Decision Tree, Random Forest, XGBoost (falls back to Gradient Boosting if not installed),
optional LSTM (`lstm_model.py`). Best model by ROC-AUC is saved automatically.

## Explainable AI
- Global: permutation feature importance (`outputs/feature_importance.png`)
- Local: occlusion attribution – each sensor is reset to a healthy value and the drop in failure probability
  ranks the "major contributing factors" for that vehicle.

## Scoring
Health % = 100 × (1 − failure probability). Status: HEALTHY < 40% ≤ WARNING < 75% ≤ CRITICAL.

## Run
    pip install -r requirements.txt
    python train.py          # trains, evaluates, saves model + charts to outputs/
    streamlit run app.py     # interactive dashboard

## Data note
No real telemetry was supplied, so `core.generate_data()` simulates fleet data (8,000 records, ~30% with
developing faults). Replace it with a real CSV (same 9 columns + `failure`) for production use; metrics
reported here reflect simulated data only.
