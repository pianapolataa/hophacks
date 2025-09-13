#dont include this is for demo?
import numpy as np

def generate_bp_hr(
    baseline_sbp, baseline_dbp, baseline_hr,
    age, sex, bmi, diabetes,
    sodium_intake, exercise_hours,
    dosage, time_of_day, hours=8
):
    """
    Generate synthetic SBP, DBP, HR trajectory for the next `hours`.
    time_of_day = current hour in [0, 24)
    """

    # --- 1. Sensitivity Model ---
    AgeFactor = max(0, (70 - age) / 70)          # younger -> more sensitive
    SexFactor = 1.1 if sex == "F" else 1.0       # females slightly more sensitive
    BMIFactor = max(0.5, 1.2 - (bmi - 25) * 0.02) # obesity reduces sensitivity
    ComorbidityFactor = 0.8 if diabetes else 1.0

    S = AgeFactor * SexFactor * BMIFactor * ComorbidityFactor
    # add noise
    S *= np.random.normal(1.0, 0.05)

    # --- 2. Circadian Rhythm ---
    def circadian_component(A, phi, t):
        return A * np.sin(2 * np.pi / 24 * (t - phi))

    A_sbp = 5 + 0.05 * (baseline_sbp - 120)   # amplitude scales with baseline
    A_dbp = 3 + 0.03 * (baseline_dbp - 80)
    A_hr  = 5 + 0.05 * (baseline_hr - 70)

    phi_sbp, phi_dbp, phi_hr = 9, 9, 15  # phases in hours

    # --- 3. Medication Effect (Lisinopril) ---
    tau = 6.0  # decay constant in hours

    def med_effect(t, dose):
        return -S * dose * np.exp(-t / tau)

    def med_effect_hr(t, dose):
        return 0.15 * med_effect(t, dose)  # weaker effect on HR

    # --- 4. Lifestyle Factors ---
    delta_Na = 0.01 * (sodium_intake / 2000) * baseline_sbp
    delta_Ex = -2.0 * exercise_hours

    # --- 5. Final Trajectories ---
    times = np.arange(0, hours + 1)  # 0 to hours
    sbp, dbp, hr = [], [], []

    for h in times:
        t = (time_of_day + h) % 24

        circ_sbp = circadian_component(A_sbp, phi_sbp, t)
        circ_dbp = circadian_component(A_dbp, phi_dbp, t)
        circ_hr  = circadian_component(A_hr,  phi_hr,  t)

        med = med_effect(h, dosage)
        med_hr = med_effect_hr(h, dosage)

        sbp_val = baseline_sbp + circ_sbp + med + delta_Na + delta_Ex + np.random.normal(0, 1)
        dbp_val = baseline_dbp + circ_dbp + 0.6 * med + delta_Na + delta_Ex + np.random.normal(0, 1)
        hr_val  = baseline_hr  + circ_hr  + med_hr - 0.5 * delta_Ex + np.random.normal(0, 1)

        sbp.append(sbp_val)
        dbp.append(dbp_val)
        hr.append(hr_val)

    return times, np.array(sbp), np.array(dbp), np.array(hr)
