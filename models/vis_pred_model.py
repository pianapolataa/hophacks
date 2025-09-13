import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pred_model import Seq2SeqBP, NUMERIC_FEATURES, CSV_PATH, load_model_artifacts, predict_patient

# ─────────── paths ───────────
CHECKPOINT_PATH = "seq2seq_bp_hr_sens_best.pt"
DATA_PATH       = "synthetic_bp_data.csv"

# ─────────── load model + scalers ───────────
model, scaler_X, scaler_Y = load_model_artifacts(CHECKPOINT_PATH)

# ─────────── load dataset ───────────
df = pd.read_csv(DATA_PATH)

# pick a random patient
pid = 817
patient_rows = df[df["patient_id"] == pid].sort_values("t_postdose")
first_row = patient_rows.iloc[0]

dose = first_row["dose"]
dose_time = int(first_row["current_time"])

# run prediction
preds, sens = predict_patient(model, scaler_X, scaler_Y, patient_rows)

# actual values from dataset
actual_seq = patient_rows[["sbp", "dbp", "hr"]].values

# ─────────── plot ───────────
time = np.arange(1, preds.shape[0] + 1)
labels = ["SBP", "DBP", "HR"]

plt.figure(figsize=(10,6))
for i, col in enumerate(["sbp_pred", "dbp_pred", "hr_pred"]):
    plt.plot(time, actual_seq[:, i], marker="o", label=f"Actual {labels[i]}")
    plt.plot(time, preds[col], linestyle="--", marker="x", label=f"Pred {labels[i]}")

plt.xlabel("Hours after dose")
plt.ylabel("Value")
plt.title(
    f"Patient {pid} — Dose={dose}mg at {dose_time}:00\nPredicted vs Actual (Sensitivity={sens:.2f})"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
