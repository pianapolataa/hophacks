from flask import Flask, render_template, request, jsonify
from optimize_dosages import optimize_patient  # your brute-force optimizer

app = Flask(__name__)

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
