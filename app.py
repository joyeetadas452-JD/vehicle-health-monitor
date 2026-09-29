"""Dashboard: streamlit run app.py"""
import joblib, pandas as pd, streamlit as st
from core import FEATURES, HEALTHY_REF, report
st.set_page_config(page_title="Vehicle Health Monitor", page_icon="🚗")
art = joblib.load("models/best_model.joblib"); model, scaler = art["model"], art["scaler"]
st.title("🚗 Predictive Maintenance & Vehicle Health")
st.caption(f"Model: {art['name']}")
rng = {"engine_temp": (60, 130, 1.0), "rpm": (500, 6000, 50.0), "oil_pressure": (10, 70, 1.0),
       "vibration": (0.0, 10.0, 0.1), "battery_voltage": (10.0, 15.0, 0.1), "coolant_temp": (60, 120, 1.0),
       "fuel_consumption": (3.0, 20.0, 0.1), "speed": (0, 160, 1.0), "operating_hours": (0, 15000, 100.0)}
vals = {}
for f in FEATURES:
    lo, hi, step = rng[f]
    vals[f] = st.slider(f.replace("_", " ").title(), float(lo), float(hi), float(HEALTHY_REF[f]), float(step))
r = report(model, scaler, pd.DataFrame([vals]))
c1, c2 = st.columns(2); c1.metric("Vehicle Health", f"{r['health']}%"); c2.metric("Failure Probability", f"{r['prob']}%")
{"HEALTHY": st.success, "WARNING": st.warning, "CRITICAL": st.error}[r["status"]](f"Status: {r['status']}")
if r["factors"]:
    st.subheader("Major contributing factors")
    for i, (n, d) in enumerate(r["factors"], 1): st.write(f"{i}. {n}  (+{d*100:.0f}% risk)")
