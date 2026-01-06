import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("dataset/heart.csv")

print("Kolom dataset:")
print(data.columns)

# =========================
# 2. Pilih Fitur & Label
# =========================
features = [
    "age",
    "sex",
    "chest pain type",
    "resting bp s",
    "cholesterol",
    "max heart rate"
]

X = data[features]
y = data["target"]

# =========================
# 3. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Scaling
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 5. Train Model
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# =========================
# 6. Evaluasi
# =========================
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# 7. Simpan Model
# =========================
joblib.dump(model, "heart_disease_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ Model dan scaler berhasil disimpan!")