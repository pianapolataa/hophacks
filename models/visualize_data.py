import pandas as pd
import matplotlib.pyplot as plt

# Load synthetic dataset
df = pd.read_csv("synthetic_bp_data.csv")

def plot_patient(pid):
    # Select data for this patient
    g = df[df['patient_id'] == pid].sort_values('t_postdose')
    t = g['t_postdose']

    # Plot SBP, DBP, HR over 8 hours
    plt.figure(figsize=(10, 5))
    plt.plot(t, g['sbp'], marker='o', label='SBP (mmHg)')
    plt.plot(t, g['dbp'], marker='o', label='DBP (mmHg)')
    plt.plot(t, g['hr'], marker='o', label='HR (bpm)')
    plt.title(f'Patient {pid} - 8h Post-dose Predictions')
    plt.xlabel('Hours after dose')
    plt.ylabel('Value')
    plt.xticks(t)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print baseline info and sensitivity
    print(f"Patient {pid} Baseline Info:")
    print(f"  Avg SBP: {g['avg_sbp'].iloc[0]:.1f} mmHg")
    print(f"  Avg DBP: {g['avg_dbp'].iloc[0]:.1f} mmHg")
    print(f"  Baseline HR: {g['baseline_hr'].iloc[0]:.1f} bpm")
    print(f"  Age: {g['age'].iloc[0]} years, Sex: {g['sex'].iloc[0]}")
    print(f"  BMI: {g['bmi'].iloc[0]:.1f}, Diabetes: {g['diabetes'].iloc[0]}")
    print(f"  Sodium Intake Today: {g['sodium_intake'].iloc[0]:.1f} mg")
    print(f"  Exercise Today: {g['exercise_today'].iloc[0]:.2f} hours")
    print(f"  Dose Given: {g['dose'].iloc[0]} mg at Hour {g['current_time'].iloc[0]}")
    print(f"  Sensitivity: {g['sensitivity'].iloc[0]:.2f}")
    print(f"  Current SBP/DBP/HR at Dose: {g['current_sbp'].iloc[0]:.1f}/{g['current_dbp'].iloc[0]:.1f}/{g['current_hr'].iloc[0]:.1f}")

# --------------------
if __name__ == "__main__":
    # Example: plot a random patient
    patient_id = df['patient_id'].sample(1).iloc[0]
    plot_patient(patient_id)
