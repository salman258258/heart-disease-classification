from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import time

app = Flask(__name__)
CORS(app)

# =========================
# Load Model & Scaler
# =========================
model = joblib.load("../model/heart_disease_model.joblib")
scaler = joblib.load("../model/scaler.joblib")

# =========================
# Monitoring Variables
# =========================
total_requests = 0
last_inference_time = 0
inference_times = []

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
    global total_requests, last_inference_time, inference_times

    start_time = time.time()
    data = request.json

    # Model hanya menggunakan 6 fitur sesuai training
    required_fields = [
        "age", "sex", "cp", "trestbps", "chol", "thalach"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({
                "error": f"Field '{field}' is required"
            }), 400

    # Ekstrak hanya 6 fitur yang sesuai dengan model
    features = [
        data["age"],
        data["sex"],
        data["cp"],
        data["trestbps"],
        data["chol"],
        data["thalach"]
    ]

    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    inference_time = time.time() - start_time

    # Monitoring
    total_requests += 1
    last_inference_time = inference_time
    inference_times.append(inference_time)

    result = (
        "Risiko Tinggi Penyakit Jantung"
        if prediction == 1
        else "Risiko Rendah Penyakit Jantung"
    )

    return jsonify({
        "prediction": int(prediction),
        "result": result,
        "inference_time": round(inference_time, 4)
    })
# =========================
# Dashboard Monitoring
# =========================
@app.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    avg_time = sum(inference_times) / len(inference_times) if inference_times else 0

    return jsonify({
        "total_requests": total_requests,
        "last_inference_time": round(last_inference_time, 4),
        "avg_inference_time": round(avg_time, 4),
        "status": "AKTIF"
    })

# =========================
# Run App
# =========================
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)