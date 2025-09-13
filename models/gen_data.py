import numpy as np
import pandas as pd
import math

def sample_baseline():
    return {
        "avg_sbp": np.random.normal(130, 15),
        "avg_dbp": np.random.normal(80, 10),
        "baseline_hr": np.random.normal(70, 8),
        "age": np.random.randint(30, 80),
        "sex": np.random.choice(["M", "F"]),
        "bmi": np.random.normal(27, 4),
        "diabetes": np.random.choice([0, 1], p=[0.7, 0.3]),
        "sodium_intake": np.random.normal(3000, 800),
        "exercise_today": np.random.exponential(0.5),
    }

# fixed set of doses (instead of random sampling)
DOSES = [0, 2.5, 5, 10, 20, 40]

def compute_sensitivity(baseline):
    sens = 1.0
    if baseline["age"] > 60: sens *= 0.85
    if baseline["bmi"] > 30: sens *= 0.9
    if baseline["diabetes"] == 1: sens *= 0.8
    if baseline["sodium_intake"] > 4000: sens *= 0.9
    if baseline["exercise_today"] > 1: sens *= 1.05
    sens = np.clip(sens * np.random.normal(1, 0.1), 0.5, 1.5)
    return sens

def circadian_factor(hour):
    bp_factor = 2 * np.sin(2 * np.pi * (hour - 9) / 24)
    hr_factor = 1.5 * np.sin(2 * np.pi * (hour - 15) / 24)
    return bp_factor, hr_factor

def drug_effect_sbp(dose, sensitivity, t):
    if dose == 0: return 0
    peak = 2.0 * dose * sensitivity
    t_peak = 1.5  # hours until max effect
    return -peak * ((t / t_peak) * np.exp(1 - t / t_peak))

def drug_effect_dbp(dose, sensitivity, t):
    if dose == 0: return 0
    peak = 2.0 * dose * sensitivity
    t_peak = 1.5
    return -peak * ((t / t_peak) * np.exp(1 - t / t_peak))

def generate_patient(pid):
    baseline = sample_baseline()
    sens = compute_sensitivity(baseline)
    dose_time = np.random.randint(0, 24)

    records = []
    for j, dose in enumerate(DOSES):
        sub_pid = pid * len(DOSES) + j  # 🔹 unique patient_id per dose
        for t in range(8):
            hour_of_day = (dose_time + t) % 24
            circ_bp, circ_hr = circadian_factor(hour_of_day)

            sbp = baseline["avg_sbp"] + 0.3*circ_bp + drug_effect_sbp(dose, sens, t) + np.random.normal(0,1.5)
            dbp = baseline["avg_dbp"] + 0.3*circ_bp + drug_effect_dbp(dose, sens, t) + np.random.normal(0,1)
            hr  = baseline["baseline_hr"] + circ_hr - 0.05*drug_effect_sbp(dose, sens, t) + np.random.normal(0,1)

            records.append({
                "patient_id": sub_pid,   # 🔹 looks just like before
                "avg_sbp": baseline["avg_sbp"],
                "avg_dbp": baseline["avg_dbp"],
                "baseline_hr": baseline["baseline_hr"],
                "age": baseline["age"],
                "sex": baseline["sex"],
                "bmi": baseline["bmi"],
                "diabetes": baseline["diabetes"],
                "sodium_intake": baseline["sodium_intake"],
                "exercise_today": baseline["exercise_today"],
                "dose": dose,
                "current_time": dose_time,
                "current_sbp": sbp,
                "current_dbp": dbp,
                "current_hr": hr,
                "sensitivity": sens,
                "sbp": sbp,
                "dbp": dbp,
                "hr": hr,
                "t_postdose": t
            })
    return records

def generate_dataset(n_patients=100):
    all_records = []
    for pid in range(n_patients):
        all_records.extend(generate_patient(pid))
    return pd.DataFrame(all_records)

def prepare_arrays(df):
    input_cols = [
        "avg_sbp","avg_dbp","baseline_hr","age","sex","bmi","diabetes",
        "sodium_intake","exercise_today","dose","current_time",
        "current_sbp","current_dbp","current_hr"
    ]
    output_cols = ["sbp","dbp","hr","sensitivity"]

    df["sex"] = (df["sex"] == "F").astype(int)

    X, y = [], []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values(["dose","t_postdose"])  # keep consistent ordering
        X.append(g[input_cols].values)
        y.append(g[output_cols].values)

    return np.array(X), np.array(y)

if __name__=="__main__":
    df = generate_dataset(n_patients=20000)
    df.to_csv("synthetic_bp_data.csv", index=False)
    print("CSV saved with shape:", df.shape)

    X, y = prepare_arrays(df)
    np.savez("synthetic_bp_data.npz", X=X, y=y)
    print("NumPy arrays saved:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
