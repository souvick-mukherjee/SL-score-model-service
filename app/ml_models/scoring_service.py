import os
import pandas as pd
import joblib
import numpy as np
import h3
from pathlib import Path

from app.ml_models.bostonscoringmodel import load_and_preprocess_data, predict_safety_score

base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "clustered_boston_h3.csv")
model_path = os.path.join(base_dir, "model", "xgb_model.pkl")

df = load_and_preprocess_data(data_path)
cluster_map = dict(zip(df['cluster'], df['safety_score']))

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")


def predict_scores(coordinates: list[dict]) -> list[dict]:
    print("Received coords:", coordinates)
    results = []
    for coord in coordinates:
        score, confidence = predict_safety_score(coord['lat'], coord['lng'], model, cluster_map, debug=False)
        results.append({
            "lat": coord['lat'],
            "lng": coord['lng'],
            "score": score,
            "confidence": confidence
        })
    print("Prediction results:", results)
    return results
