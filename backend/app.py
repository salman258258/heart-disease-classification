from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import time

app = Flask(__name__)
CORS(app)

model = joblib.load("../model/heart_disease_model.joblib")
scaler = joblib.load("../model/scaler.joblib")

# =========================
# Health Check
# =========================
@app.route("/")
def home():
    return "Heart Disease Prediction API is running"

# =========================
# Predict Endpoint
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    data = request.json

    features = [
        data["age"],
        data["sex"],
        data["chest_pain_type"],
        data["resting_bp"],
        data["cholesterol"],
        data["max_heart_rate"]
    ]

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    inference_time = time.time() - start_time

    result = "Risiko Tinggi Penyakit Jantung" if prediction == 1 else "Risiko Rendah Penyakit Jantung"

    return jsonify({
        "prediction": int(prediction),
        "result": result,
        "inference_time": round(inference_time, 4)
    })


if __name__ == "__main__":
    app.run(debug=True)