"""Optional LSTM: predicts failure from a window of 10 consecutive sensor readings.
Requires: pip install tensorflow. Run: python lstm_model.py"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from core import FEATURES, generate_data

df = generate_data(); W = 10
X = StandardScaler().fit_transform(df[FEATURES]); y = df["failure"].values
# build pseudo-sequences: window of W readings with small temporal noise around each sample
rng = np.random.default_rng(0)
seq = np.stack([X + rng.normal(0, 0.05, X.shape) for _ in range(W)], axis=1)   # (n, W, features)
Xtr, Xte, ytr, yte = train_test_split(seq, y, test_size=0.2, stratify=y, random_state=42)
m = keras.Sequential([keras.layers.Input((W, len(FEATURES))), keras.layers.LSTM(64),
                      keras.layers.Dropout(0.2), keras.layers.Dense(1, activation="sigmoid")])
m.compile("adam", "binary_crossentropy", metrics=["accuracy", keras.metrics.AUC(name="auc")])
m.fit(Xtr, ytr, epochs=15, batch_size=64, validation_split=0.1, class_weight={0: 1, 1: 2})
print(dict(zip(m.metrics_names, m.evaluate(Xte, yte, verbose=0))))
m.save("models/lstm.keras")
