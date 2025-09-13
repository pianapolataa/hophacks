# brute_force_search.py
import pandas as pd
import numpy as np
from itertools import product
from use_model import predict_for_input  # import from your file
from use_model import load_model_and_scalers, DEVICE

def compute_reward(sbp_traj, dbp_traj, hr_traj, patient):
    sbp_penalty = np.sum((np.maximum(0, sbp_traj - 130))**2)
    dbp_penalty = np.sum((np.maximum(0, dbp_traj - 80))**2)
    hypo_penalty = 5 * np.sum((np.maximum(0, 90 - sbp_traj))**2 +
                              (np.maximum(0, 60 - dbp_traj))**2)
    hr_penalty = 0.1 * np.sum((hr_traj - patient["baseline_hr"])**2)

    dose_penalty = 0.01 * (patient["dose"]**2)
    sodium_penalty = 0.001 * ((patient["sodium_intake"] - 3000)**2)
    exercise_penalty = 0.05 * (patient["exercise_today"]**2)

    reward = -(sbp_penalty + dbp_penalty + hypo_penalty +
               hr_penalty + dose_penalty + sodium_penalty + exercise_penalty)
    return reward

dose_grid = [0, 2.5, 5, 10, 20, 40]
hour_grid = [8, 9, 10, 11, 12, 13, 14,15, 16, 17, 18, 19, 20, 21, 22]
sodium_grid = list(range(0, 2001, 100))  # delta sodium
exercise_grid = [0.5, 1.0, 1.5, 2.0]     # delta exercise
TOP_K = 5
MODEL_PATH = "models/seq2seq_bp_hr_sens_best.pt"
model, scX, scY = load_model_and_scalers(MODEL_PATH)
model.eval()

def optimize_patient(baseline_info: dict):
    results = []
    print(baseline_info)
    for dose, hour, delta_sodium, extra_ex in product(dose_grid, hour_grid, sodium_grid, exercise_grid):
        # copy baseline info
        patient = baseline_info.copy()
        patient["dose"] = dose
        patient["current_time"] = hour
        patient["sodium_intake"] += delta_sodium
        patient["exercise_today"] += extra_ex

        # predict next 8h BP/HR
        preds_df, _ = predict_for_input(model, scX, scY, patient)

        # compute reward
        reward = compute_reward(preds_df["sbp_pred"].values,
                                preds_df["dbp_pred"].values,
                                preds_df["hr_pred"].values,
                                patient)
        results.append({
            "dose": dose,
            "hour": hour,
            "delta_sodium": delta_sodium,
            "delta_exercise": extra_ex,
            "reward": reward
        })

    # sort top K
    top_results = sorted(results, key=lambda x: x["reward"], reverse=True)[:TOP_K]
    return top_results

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # user baseline input
    baseline_info = {
        "avg_sbp": 135,
        "avg_dbp": 85,
        "baseline_hr": 72,
        "age": 20,
        "sex": "F",
        "bmi": 28,
        "diabetes": 0,
        "sodium_intake": 2500,
        "exercise_today": 0.5
    }

    # ---------------- BRUTE-FORCE ----------------
    top_actions = optimize_patient(baseline_info)
    print("Top action combos:")
    for i, combo in enumerate(top_actions, 1):
        print(f"Rank {i}: {combo}")

    # ---------------- TAKE TOP 1 ----------------
    best_action = top_actions[0]
    patient_for_plot = baseline_info.copy()
    patient_for_plot["dose"] = best_action["dose"]
    patient_for_plot["current_time"] = best_action["hour"]
    patient_for_plot["sodium_intake"] += best_action["delta_sodium"]
    patient_for_plot["exercise_today"] += best_action["delta_exercise"]

    preds_df, sens = predict_for_input(model, scX, scY, patient_for_plot)

    # ---------------- PLOT ----------------
    plt.figure(figsize=(10,5))
    plt.plot(preds_df["hour_offset"], preds_df["sbp_pred"], marker='o', label="SBP")
    plt.plot(preds_df["hour_offset"], preds_df["dbp_pred"], marker='o', label="DBP")
    plt.plot(preds_df["hour_offset"], preds_df["hr_pred"], marker='o', label="HR")
    plt.xticks(preds_df["hour_offset"])
    plt.xlabel("Hour offset")
    plt.ylabel("Value")
    plt.title(f"Predicted next 8 hours (Top action reward={best_action['reward']:.1f})")
    plt.legend()
    plt.grid(True)
    plt.show()
