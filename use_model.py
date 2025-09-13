# use_model.py
"""
Standalone inference script for your Seq2SeqBP model.
Place this file next to the `models/` folder that contains:
  - seq2seq_bp_hr_sens_best.pt
  - scaler_X.joblib
  - scaler_Yseq.joblib

This script:
 - Recreates the model architecture exactly as in training,
 - Loads checkpoint and scalers (resolving scaler paths relative to the checkpoint),
 - Provides predict_patient(...) to produce next-8-hour SBP/DBP/HR and sensitivity.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple

# ---------------- config that matches training ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 8

NUMERIC_FEATURES = [
    "avg_sbp","avg_dbp","baseline_hr","age","bmi","diabetes",
    "sodium_intake","exercise_today","dose","current_time",
    "current_sbp","current_dbp","current_hr"
]   # 13 numeric features in training order
# training appended sex flag as the last feature (1.0 if sex == "F" else 0.0)

# ---------------- Model definitions (copy from training) ----------
class Encoder(nn.Module):
    def __init__(self, inp, hid, lat):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inp,hid), nn.ReLU(),
                                 nn.Linear(hid,lat), nn.ReLU())
    def forward(self,x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, inp, hid, outp, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(inp,hid,n_layers,batch_first=True)
        self.fc   = nn.Linear(hid,outp)
    def forward(self,x,h): o,(hn,cn)=self.lstm(x,h); return self.fc(o),(hn,cn)

class Seq2SeqBP(nn.Module):
    def __init__(self, in_dim, lat=64, enc_h=128, dec_in=3,
                 dec_h=128, out_dim=3, n_layers=1):
        super().__init__()
        self.encoder = Encoder(in_dim, enc_h, lat)
        self.decoder = Decoder(dec_in, dec_h, out_dim, n_layers)
        self.h0_proj = nn.Linear(lat, dec_h)
        self.c0_proj = nn.Linear(lat, dec_h)
        self.sens_head = nn.Sequential(
            nn.Linear(lat, max(8, lat//2)), nn.ReLU(),
            nn.Linear(max(8, lat//2), 1), nn.Sigmoid())

    def forward(self,src,dec_init,trg_seq=None,tf_ratio=0.5):
        lat   = self.encoder(src)
        sens  = self.sens_head(lat)
        h0    = torch.tanh(self.h0_proj(lat)).unsqueeze(0)
        c0    = torch.tanh(self.c0_proj(lat)).unsqueeze(0)
        hidden= (h0,c0)

        out_seq=[]
        dec_in = dec_init
        for t in range(SEQ_LEN):
            pred,hidden = self.decoder(dec_in,hidden)   # (B,1,3)
            out_seq.append(pred)
            use_teacher = trg_seq is not None and float(torch.rand(1).item()) < tf_ratio
            dec_in = trg_seq[:,t:t+1] if use_teacher else pred.detach()
        out_seq = torch.cat(out_seq, dim=1)             # (B,8,3)
        return out_seq, sens

# ---------------- utility: robust checkpoint + scaler loader ---------------
def load_model_and_scalers(model_path: str):
    """
    Loads model state_dict and scalers from model_path.
    Handles checkpoints that contain:
      {"model_state": model.state_dict(),
       "scaler_X": "scaler_X.joblib",
       "scaler_Y": "scaler_Yseq.joblib"}
    and resolves scaler paths relative to the model_path directory.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=DEVICE)

    # determine whether checkpoint stored a nested dict or raw state_dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # assume ckpt is itself a state_dict
        state_dict = ckpt

    # find scalers: ckpt may contain string paths for scaler filenames
    base_dir = os.path.dirname(os.path.abspath(model_path))
    scX = scY = None
    if isinstance(ckpt, dict):
        # handle keys storing scaler *paths* or scaler objects directly
        for key in ("scaler_X", "scalerX", "scX"):
            if key in ckpt:
                val = ckpt[key]
                if isinstance(val, str):
                    candidate = os.path.join(base_dir, val)
                    if os.path.exists(candidate):
                        scX = joblib.load(candidate)
                    elif os.path.exists(val):
                        scX = joblib.load(val)
                    else:
                        raise FileNotFoundError(f"Scaler X path in checkpoint not found: {val} (tried {candidate})")
                else:
                    # assume scaler object stored directly
                    scX = val
                break

        for key in ("scaler_Y", "scalerY", "scY", "scaler_Yseq"):
            if key in ckpt:
                val = ckpt[key]
                if isinstance(val, str):
                    candidate = os.path.join(base_dir, val)
                    if os.path.exists(candidate):
                        scY = joblib.load(candidate)
                    elif os.path.exists(val):
                        scY = joblib.load(val)
                    else:
                        raise FileNotFoundError(f"Scaler Y path in checkpoint not found: {val} (tried {candidate})")
                else:
                    scY = val
                break

    if scX is None or scY is None:
        # still allow loading even if scalers are missing: warn the user
        print("Warning: Could not find scalers in checkpoint. Predictions will be made without scaling.")
    else:
        # ensure they look like StandardScaler (have .transform/.inverse_transform and .mean_)
        if not (hasattr(scX, "transform") and hasattr(scX, "mean_")):
            print("Warning: loaded scX does not look like a sklearn StandardScaler.")
        if not (hasattr(scY, "transform") and hasattr(scY, "inverse_transform")):
            print("Warning: loaded scY does not look like a sklearn StandardScaler for sequences.")

    # Instantiate model with the exact in_dim used at training time
    if scX is not None:
        in_dim = len(scX.mean_)
    else:
        # fallback: in_dim = len(NUMERIC_FEATURES) + 1 (sex flag appended)
        in_dim = len(NUMERIC_FEATURES) + 1

    model = Seq2SeqBP(in_dim=in_dim).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scX, scY

# ---------------- prediction utility (compatible with your training) -----
def _encode_sex(raw_sex):
    """Return 1.0 for female, 0.0 for non-female (behaves like training code)."""
    # training did: feats.append(1.0 if first["sex"]=="F" else 0.0)
    if raw_sex is None:
        return 0.0
    if isinstance(raw_sex, str):
        s = raw_sex.strip().upper()
        return 1.0 if s == "F" or s == "FEMALE" else 0.0
    # numeric -> assume 1 => female, 0 => male
    try:
        val = float(raw_sex)
        return 1.0 if val == 1.0 else 0.0
    except Exception:
        return 0.0

def predict_patient(model: Seq2SeqBP, scX, scY, patient_rows: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    patient_rows: pd.DataFrame containing the same columns used during training,
                  grouped rows for a patient (sorted by t_postdose). Only the
                  first row is used for static inputs (matching BPSeqDataset).
    Returns (df_out, sens_val)
    df_out: dataframe with hour_offset, target_hour, sbp_pred, dbp_pred, hr_pred
    sens_val: float in 0..1
    """
    if not isinstance(patient_rows, pd.DataFrame):
        raise ValueError("patient_rows must be a pandas DataFrame")

    # must have at least one row
    if len(patient_rows) == 0:
        raise ValueError("patient_rows is empty")

    g = patient_rows.sort_values("t_postdose")
    first = g.iloc[0]

    # build feature vector in exact NUMERIC_FEATURES order, then append sex flag
    feats = []
    for col in NUMERIC_FEATURES:
        if col not in first:
            raise KeyError(f"Required column '{col}' missing from patient_rows")
        feats.append(first[col])

    # sex may be stored as 'M'/'F' or 0/1 - training converted 'F' -> 1.0 else 0.0
    sex_raw = first.get("sex", None)
    sex_flag = _encode_sex(sex_raw)
    feats.append(sex_flag)

    feats_arr = np.array(feats, dtype=np.float32).reshape(1, -1)

    # apply X scaler if available
    if scX is not None:
        X_scaled = scX.transform(feats_arr).astype(np.float32)
    else:
        X_scaled = feats_arr.astype(np.float32)

    X = torch.tensor(X_scaled, device=DEVICE)

    # indices in the original NUMERIC_FEATURES block
    sbp_i = NUMERIC_FEATURES.index("current_sbp")
    dbp_i = NUMERIC_FEATURES.index("current_dbp")
    hr_i  = NUMERIC_FEATURES.index("current_hr")

    # when we appended sex flag the feature vector length = len(NUMERIC_FEATURES)+1
    # X is shape (1, in_dim)
    # select scaled current_sbp/current_dbp/current_hr -> create dec_init shape (1,1,3)
    dec_init = torch.cat([
        X[:, sbp_i:sbp_i+1],
        X[:, dbp_i:dbp_i+1],
        X[:, hr_i:hr_i+1]
    ], dim=1).unsqueeze(1)   # shape (1,1,3)

    # run model in autoregressive mode
    seq_scaled_torch, sens_torch = model(X, dec_init, trg_seq=None, tf_ratio=0.0)
    # seq_scaled_torch shape: (1,8,3)
    seq_scaled_np = seq_scaled_torch.detach().cpu().numpy().reshape(1, -1)  # detach before numpy

    # inverse transform if scaler Y exists
    if scY is not None:
        seq_np = scY.inverse_transform(seq_scaled_np).reshape(SEQ_LEN, 3)
    else:
        # no scaler: assume predictions already in natural units
        seq_np = seq_scaled_np.reshape(SEQ_LEN, 3)

    sens_val = float(sens_torch.cpu().item())

    base_hour = int(first["current_time"]) if "current_time" in first else 0
    df_out = pd.DataFrame({
        "hour_offset": np.arange(1, SEQ_LEN+1),
        "target_hour": [ (base_hour + i) % 24 for i in range(1, SEQ_LEN+1) ],
        "sbp_pred": seq_np[:, 0],
        "dbp_pred": seq_np[:, 1],
        "hr_pred":  seq_np[:, 2]
    })
    return df_out, sens_val

# ------------------- Helper to predict from a dictionary -----------------
def predict_for_input(model, scX, scY, input_features: dict) -> Tuple[pd.DataFrame, float]:
    """
    input_features: dict with any of the training features:
        NUMERIC_FEATURES + 'sex' + 'patient_id' + 't_postdose'
    Missing keys are filled with default values from example patient_data.
    Returns (preds_df, sensitivity)
    """
    # default values
    defaults = {
        "patient_id": 0,
        "avg_sbp": 135.0, "avg_dbp": 85.0, "baseline_hr": 72.0,
        "age": 58, "sex": 1, "bmi": 27.5, "diabetes": 0,
        "sodium_intake": 2400.0, "exercise_today": 0.5,
        "dose": 10.0, "current_time": 14,
        "current_sbp": 140.0, "current_dbp": 90.0, "current_hr": 75.0,
        "sensitivity": 0.0, "sbp": 0, "dbp": 0, "hr": 0, "t_postdose": 0
    }

    # merge defaults with user input (input_features overrides defaults)
    patient_input = {**defaults, **input_features}
    patient_df = pd.DataFrame([patient_input])

    # call existing predict_patient
    return predict_patient(model, scX, scY, patient_df)

# ------------------- Example usage as script -----------------------
if __name__ == "__main__":
    # change this if your model path differs
    MODEL_PATH = os.path.join("models", "seq2seq_bp_hr_sens_best.pt")

    try:
        model, scX, scY = load_model_and_scalers(MODEL_PATH)
    except Exception as e:
        print("Error loading model/scalers:", e)
        raise

    # Example patient; make sure fields match training names exactly.
    patient_data = {
        "patient_id": 0,
        "avg_sbp": 135.0,
        "avg_dbp": 85.0,
        "baseline_hr": 72.0,
        "age": 58,                # <- fix any accidental huge numbers
        "sex": 1,               # "M" or "F" or numeric 0/1
        "bmi": 27.5,
        "diabetes": 0,
        "sodium_intake": 2400.0,
        "exercise_today": 0.5,
        "dose": 10.0,
        "current_time": 14,
        "current_sbp": 140.0,
        "current_dbp": 90.0,
        "current_hr": 75.0,
        "sensitivity": 0.0,   # placeholder, not used for prediction
        "sbp": 0, "dbp": 0, "hr": 0,
        "t_postdose": 0
    }

    # sanity check for weird inputs
    if patient_data["age"] > 120 or patient_data["age"] < 0:
        print(f"Warning: suspicious age value: {patient_data['age']}")

    patient_df = pd.DataFrame([patient_data])

    preds_df, sens = predict_patient(model, scX, scY, patient_df)
    print("\nPredicted next 8 hours:")
    print(preds_df.to_string(index=False))
    print(f"Sensitivity (0-1): {sens:.3f}")

    preds_df, sens = predict_for_input(model, scX, scY, {
        "avg_sbp": 130,
        "avg_dbp": 80,
        "dose": 5,
        "sex": "F",
        "age": 65
    })

    print(preds_df)
    print("Predicted sensitivity:", sens)

