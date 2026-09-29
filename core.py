"""Shared logic: synthetic sensor data, preprocessing, health score, XAI, alerts."""
import numpy as np, pandas as pd

FEATURES = ["engine_temp", "rpm", "oil_pressure", "vibration", "battery_voltage",
            "coolant_temp", "fuel_consumption", "speed", "operating_hours"]
LABELS = {"engine_temp": "Rising engine temperature", "rpm": "Abnormal RPM",
          "oil_pressure": "Abnormal oil pressure", "vibration": "High vibration",
          "battery_voltage": "Low battery voltage", "coolant_temp": "High coolant temperature",
          "fuel_consumption": "High fuel consumption", "speed": "Abnormal speed profile",
          "operating_hours": "High operating hours"}
HEALTHY_REF = {"engine_temp": 90, "rpm": 2500, "oil_pressure": 45, "vibration": 2.0,
               "battery_voltage": 13.8, "coolant_temp": 88, "fuel_consumption": 8.0,
               "speed": 70, "operating_hours": 3000}

def generate_data(n=8000, seed=42):
    """Simulate fleet telemetry. Failure risk grows with wear (hours) and sensor drift."""
    r = np.random.default_rng(seed)
    hours = r.uniform(100, 12000, n)
    wear = hours / 12000
    fault = r.random(n) < 0.30                      # vehicles with a developing fault
    sev = np.where(fault, r.uniform(0.3, 1.0, n), 0)
    df = pd.DataFrame({
        "engine_temp": r.normal(90, 4, n) + 18*sev + 6*wear,
        "rpm": r.normal(2500, 350, n) + 300*sev*r.choice([-1, 1], n),
        "oil_pressure": r.normal(45, 4, n) - 14*sev*r.choice([0.6, 1], n) - 4*wear,
        "vibration": np.abs(r.normal(2.0, 0.5, n)) + 3.2*sev + 1.2*wear,
        "battery_voltage": r.normal(13.8, 0.3, n) - 1.1*sev*r.random(n) - 0.4*wear,
        "coolant_temp": r.normal(88, 4, n) + 14*sev + 4*wear,
        "fuel_consumption": r.normal(8, 1, n) + 2.5*sev + 1.0*wear,
        "speed": np.clip(r.normal(70, 20, n), 0, 140),
        "operating_hours": hours})
    logit = -4.2 + 5.0*sev + 2.4*wear + r.normal(0, 0.6, n)
    p = 1 / (1 + np.exp(-logit))
    df["failure"] = (r.random(n) < p).astype(int)
    return df

def health_score(prob):
    """Vehicle health % = inverse of failure probability."""
    return round(float(100 * (1 - prob)), 1)

def status(prob):
    return "CRITICAL" if prob >= 0.75 else "WARNING" if prob >= 0.40 else "HEALTHY"

def contributing_factors(model, scaler, row: pd.DataFrame, top=3):
    """Model-agnostic local XAI: replace each sensor with its healthy reference value and
    measure the drop in failure probability (occlusion attribution)."""
    base = model.predict_proba(scaler.transform(row[FEATURES]))[0, 1]
    out = []
    for f in FEATURES:
        r2 = row[FEATURES].copy(); r2[f] = HEALTHY_REF[f]
        d = base - model.predict_proba(scaler.transform(r2))[0, 1]
        out.append((f, d))
    out.sort(key=lambda x: -x[1])
    return [(LABELS[f], round(d, 3)) for f, d in out[:top] if d > 0.005]

def report(model, scaler, row):
    p = float(model.predict_proba(scaler.transform(row[FEATURES]))[0, 1])
    return {"health": health_score(p), "prob": round(100*p, 1), "status": status(p),
            "factors": contributing_factors(model, scaler, row)}
