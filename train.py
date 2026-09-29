"""Train & compare models, save best, produce charts. Run: python train.py"""
import joblib, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from core import FEATURES, generate_data, report

df = generate_data(); df.to_csv("outputs/vehicle_sensor_data.csv", index=False)
X, y = df[FEATURES], df["failure"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

models = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
          "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
          "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)}
try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                                      scale_pos_weight=(ytr==0).sum()/(ytr==1).sum(), eval_metric="logloss")
except ImportError:
    models["Gradient Boosting (XGBoost substitute)"] = GradientBoostingClassifier(random_state=42)

rows, fitted = [], {}
for n, m in models.items():
    m.fit(Xtr_s, ytr); fitted[n] = m
    pr = m.predict_proba(Xte_s)[:, 1]; pd_ = (pr >= 0.5).astype(int)
    rows.append({"Model": n, "Accuracy": accuracy_score(yte, pd_), "Precision": precision_score(yte, pd_),
                 "Recall": recall_score(yte, pd_), "F1": f1_score(yte, pd_), "ROC-AUC": roc_auc_score(yte, pr)})
res = pd.DataFrame(rows).round(3).sort_values("ROC-AUC", ascending=False)
res.to_csv("outputs/model_comparison.csv", index=False); print(res.to_string(index=False))

best_name = res.iloc[0]["Model"]; best = fitted[best_name]
joblib.dump({"model": best, "scaler": sc, "name": best_name}, "models/best_model.joblib")

# Charts
plt.figure(figsize=(6, 5))
for n, m in fitted.items():
    fpr, tpr, _ = roc_curve(yte, m.predict_proba(Xte_s)[:, 1]); plt.plot(fpr, tpr, label=n)
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("outputs/roc_curves.png", dpi=150); plt.close()

ConfusionMatrixDisplay(confusion_matrix(yte, best.predict(Xte_s)), display_labels=["No failure", "Failure"]).plot(cmap="Blues")
plt.title(f"Confusion matrix – {best_name}"); plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=150); plt.close()

pi = permutation_importance(best, Xte_s, yte, n_repeats=10, random_state=42, scoring="roc_auc")
imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
imp.plot.barh(color="#2a6fdb", figsize=(6, 4)); plt.title("Global feature importance (XAI)")
plt.tight_layout(); plt.savefig("outputs/feature_importance.png", dpi=150); plt.close()

# Demo prediction on a stressed vehicle
demo = pd.DataFrame([{"engine_temp": 98, "rpm": 2700, "oil_pressure": 38, "vibration": 3.9,
                      "battery_voltage": 13.3, "coolant_temp": 93, "fuel_consumption": 9.0,
                      "speed": 72, "operating_hours": 7000}])
rep = report(best, sc, demo); print("\nSample output:", json.dumps(rep, indent=1))
json.dump(rep, open("outputs/sample_output.json", "w"), indent=1)
