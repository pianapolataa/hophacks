from flask import Flask, render_template, request, jsonify
from optimize_dosages import optimize_patient  # your brute-force optimizer
from use_model import predict_for_input, load_model_and_scalers

MODEL_PATH = "models/seq2seq_bp_hr_sens_best.pt"
model, scX, scY = load_model_and_scalers(MODEL_PATH)
model.eval()
app = Flask(__name__)

@app.route("/api/predict_experiment", methods=["POST"])
def predict_experiment():
    """
    Expects JSON:
    {
        "dosage": 10.0,
        "time_of_day": 14,
        "patient_data": { ... }  # optional, from localStorage
    }
    Returns JSON:
    {
        "predictions": [...],  # list of dicts with hour_offset, target_hour, sbp_pred, dbp_pred, hr_pred
        "sensitivity": float
    }
    """
    data = request.json
    dosage = data.get("dosage")
    time_of_day = data.get("time_of_day")
    patient_data = data.get("patient_data", {})

    if dosage is None or time_of_day is None:
        return jsonify({"error": "Dosage and time_of_day are required"}), 400

    # Merge user input with stored patient data
    patient_input = {**patient_data, "dose": dosage, "current_time": time_of_day}

    try:
        preds_df, sens = predict_for_input(model, scX, scY, patient_input)
        preds_json = preds_df.to_dict(orient="records")
        return jsonify({"predictions": preds_json, "sensitivity": sens})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/data")
def data():
    return render_template("data.html")

@app.route("/experiment")
def experiment():
    return render_template("experiment.html")

@app.route("/effects")
def effects():
    return render_template("effects.html")

@app.route("/suggestions")
def suggestions():
    return render_template("suggestions.html")


# -------------------- API endpoint for frontend --------------------
@app.route("/api/get_suggestions", methods=["POST"])
def api_get_suggestions():
    """
    Expects JSON with patient input features.
    Returns top 5 dosage suggestions + top 1 prediction for graphing.
    """
    baseline_info = request.json
    print(baseline_info)
    # Run optimizer (it loads model internally)
    top_5 = optimize_patient(baseline_info)

    # Take top 1 action for plotting
    best_action = top_5[0]
    patient_for_plot = baseline_info.copy()
    patient_for_plot["dose"] = best_action["dose"]
    patient_for_plot["current_time"] = best_action["hour"]
    patient_for_plot["sodium_intake"] += best_action["delta_sodium"]
    patient_for_plot["exercise_today"] += best_action["delta_exercise"]

    from use_model import predict_for_input, load_model_and_scalers
    MODEL_PATH = "models/seq2seq_bp_hr_sens_best.pt"
    model, scX, scY = load_model_and_scalers(MODEL_PATH)
    model.eval()
    top_1_pred_df, _ = predict_for_input(model, scX, scY, patient_for_plot)

    # Convert prediction DataFrame to JSON-friendly format
    top_1_pred = top_1_pred_df.to_dict(orient="records")

    return jsonify({
        "top_5": top_5,
        "top_1_prediction": top_1_pred
    })


if __name__ == "__main__":
    app.run(debug=True)
