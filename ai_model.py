"""
Module 3: AI Detection Module
Trains an Isolation Forest model and uses it for anomaly detection.
Run this file first: python ai_model.py
"""

import os
import json
import random
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from feature_extractor import (
    FEATURE_COLUMNS,
    simulate_packet,
    extract_features_from_packet,
    get_feature_vector,
)

MODEL_PATH  = "isolation_forest_model.pkl"
SCALER_PATH = "scaler.pkl"


# ─────────────────────────────────────────
# 1. Generate synthetic training data
# ─────────────────────────────────────────

def generate_training_data(n_normal: int = 2000, n_attack: int = 1000):
    """Generate labelled synthetic traffic for training."""
    rows, labels = [], []

    for _ in range(n_normal):
        pkt = simulate_packet(is_attack=False)
        rows.append(get_feature_vector(pkt))
        labels.append(0)   # normal

    for _ in range(n_attack):
        pkt = simulate_packet(is_attack=True)
        rows.append(get_feature_vector(pkt))
        labels.append(1)   # anomaly

    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df["label"] = labels
    return df


# ─────────────────────────────────────────
# 2. Train the model
# ─────────────────────────────────────────

def train_model():
    print("[AI] Generating synthetic training data …")
    df = generate_training_data()

    X = df[FEATURE_COLUMNS].values

    print("[AI] Scaling features …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[AI] Training Isolation Forest …")
    model = IsolationForest(
        n_estimators=300,
        contamination=0.30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[AI] Model saved → {MODEL_PATH}")
    print(f"[AI] Scaler saved → {SCALER_PATH}")

    # Quick evaluation on training set
    preds  = model.predict(X_scaled)          # +1 normal, -1 anomaly
    labels = df["label"].values               # 0 normal,  1 anomaly
    mapped = np.where(preds == -1, 1, 0)

    print("\n[AI] Evaluation on training data:")
    print(classification_report(labels, mapped, target_names=["Normal", "Anomaly"]))
    return model, scaler


# ─────────────────────────────────────────
# 3. Load saved model
# ─────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[AI] No saved model found — training now …")
        return train_model()
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[AI] Model loaded from disk.")
    return model, scaler


# ─────────────────────────────────────────
# 4. Predict on a single packet
# ─────────────────────────────────────────

def predict_packet(packet_data: dict, model, scaler) -> dict:
    """
    Returns a prediction dict:
      is_anomaly   – bool
      anomaly_score – float (higher = more anomalous; range roughly 0-1)
      severity      – str  (LOW / MEDIUM / HIGH)
      description   – str
    """
    features = get_feature_vector(packet_data)
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred  = model.predict(X_scaled)[0]          # +1 or -1
    score = model.score_samples(X_scaled)[0]    # negative log-likelihood; more negative = anomalous

    # Convert score to 0-1 range (approx)
    normalised_score = max(0.0, min(1.0, -score / 0.5))

    is_anomaly = pred == -1

    if normalised_score > 0.75:
        severity = "HIGH"
    elif normalised_score > 0.45:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    description = _build_description(packet_data, is_anomaly, severity)

    return {
        "is_anomaly":     is_anomaly,
        "anomaly_score":  round(normalised_score, 4),
        "severity":       severity if is_anomaly else "NONE",
        "description":    description,
        "features":       dict(zip(FEATURE_COLUMNS, features)),
    }


def _build_description(pkt: dict, is_anomaly: bool, severity: str) -> str:
    if not is_anomaly:
        return "Normal traffic pattern."

    reasons = []
    if int(pkt.get("ttl", 64)) < 32:
        reasons.append("abnormally low TTL")
    if int(pkt.get("packet_size", 0)) > 1300:
        reasons.append("oversized packet")
    if int(pkt.get("dst_port", 0)) in [22, 23, 3389, 445]:
        reasons.append(f"suspicious port {pkt.get('dst_port')}")
    flags = pkt.get("flags", "")
    if "R" in flags and "S" in flags:
        reasons.append("RST+SYN flag combination")

    if reasons:
        return f"{severity} anomaly detected: " + ", ".join(reasons) + "."
    return f"{severity} anomaly detected: unusual traffic pattern."


# ─────────────────────────────────────────
# 5. Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    model, scaler = train_model()

    print("\n[AI] Testing on 5 normal + 5 attack packets …")
    for i in range(5):
        pkt    = simulate_packet(is_attack=False)
        result = predict_packet(pkt, model, scaler)
        label  = "ANOMALY" if result["is_anomaly"] else "NORMAL"
        print(f"  Normal   pkt {i+1}: {label} | score={result['anomaly_score']}")

    for i in range(5):
        pkt    = simulate_packet(is_attack=True)
        result = predict_packet(pkt, model, scaler)
        label  = "ANOMALY" if result["is_anomaly"] else "NORMAL"
        print(f"  Attack   pkt {i+1}: {label} | score={result['anomaly_score']} | {result['severity']}")