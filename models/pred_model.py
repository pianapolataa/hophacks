"""
Seq2Seq LSTM (PyTorch) prototype — TRAINING WITH SYNTHETIC DATA
────────────────────────────────────────────────────────────────
New in this revision
🆕 1. The model now **also predicts heart-rate (HR)** for every future hour,
      so the decoder outputs 3 values per step:  [SBP, DBP, HR].
🆕 2. The synthetic generator creates `hr_1 … hr_8` targets.
🆕 3. `dec_init` uses the current **SBP, DBP, HR** (scaled) as first decoder input.
🆕 4. Sequence shapes change from (seq_len, 2) to (seq_len, 3).
     All scalers, losses, and helper functions updated accordingly.

Other features from the previous version are kept:
• Sensitivity in the 0–1 range (sigmoid head, no scaler).
• Still “prototype only”, *not for clinical use*.

"""

import random, math, joblib, os
from typing import Dict
import numpy as np, pandas as pd

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ───────────────────── CONFIG ──────────────────────
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 8
BATCH_SIZE, EPOCHS, LR = 64, 200, 1e-3
TEACHER_FORCING_RATIO = .5
RANDOM_SEED = 42
SENS_WEIGHT = 1.0

MODEL_OUT        = "seq2seq_bp_hr_sens_best.pt"
SCALER_X_PATH    = "scaler_X_synth.joblib"
SCALER_Y_SEQ_PATH= "scaler_y_seq_synth.joblib"

# ───────────────────── REPRO ────────────────────────
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
seed_everything()

# ─────────────── FEATURE / TARGET NAMES ─────────────
NUMERIC_FEATURES = [
    "avg_day_sbp","avg_day_dbp","baseline_hr","age","bmi",
    "diabetes","kidney","sodium_mg_today","exercise_hours_today",
    "lisinopril_mg_now","current_time_hour","sbp_now","dbp_now","hr_now"
]
SEQ_TARGET_SBPS = [f"sbp_{i}" for i in range(1, SEQ_LEN+1)]
SEQ_TARGET_DBPS = [f"dbp_{i}" for i in range(1, SEQ_LEN+1)]
SEQ_TARGET_HRS  = [f"hr_{i}"  for i in range(1, SEQ_LEN+1)]
SENS_COL = "sensitivity"

# ────────────── SYNTHETIC DATA ──────────────────────
def generate_synthetic_dataframe(n_samples=50_000, seed=RANDOM_SEED)->pd.DataFrame:
    rng, rows = np.random.RandomState(seed), []
    for _ in range(n_samples):
        avg_day_sbp = rng.normal(125,10); avg_day_dbp = rng.normal(77,6)
        baseline_hr  = rng.normal(70,8);  age = int(rng.randint(18,90))
        sex = rng.choice(['M','F']); bmi  = max(15., rng.normal(26,4))
        diabetes = int(rng.binomial(1,.12)); kidney = int(rng.binomial(1,.05))
        sodium_mg_today = max(0., rng.normal(2500,800))
        exercise_hours_today = float(max(0., rng.exponential(.25)))
        lisinopril_mg_now = float(rng.choice([0,2.5,5,10,20], p=[.6,.1,.15,.1,.05]))
        current_time_hour = int(rng.randint(0,24))

        circadian_now = 5*math.sin(2*math.pi*current_time_hour/24)
        sbp_now = avg_day_sbp + circadian_now + rng.normal(0,4) - lisinopril_mg_now*.6
        dbp_now = avg_day_dbp + circadian_now*.4+ rng.normal(0,3) - lisinopril_mg_now*.35
        hr_now  = baseline_hr + rng.normal(0,5) + exercise_hours_today*5

        sbp_future, dbp_future, hr_future = [], [], []
        for h in range(1,SEQ_LEN+1):
            circ = 5*math.sin(2*math.pi*(current_time_hour+h)/24)
            drug_eff = lisinopril_mg_now*.6 * math.exp(-((h-3)**2)/(2*2.**2))
            sbp_f = avg_day_sbp + circ - drug_eff + rng.normal(0,4)+diabetes*3+kidney*4
            dbp_f = avg_day_dbp + circ*.4- drug_eff*.6 + rng.normal(0,3)+diabetes*2+kidney*3
            hr_f  = baseline_hr + circ*.2 + rng.normal(0,5) + exercise_hours_today*5*math.exp(-h/4)
            sbp_future.append(sbp_f); dbp_future.append(dbp_f); hr_future.append(hr_f)

        # ─── 0-1 sensitivity ──────────────────────
        base_sens = .5 + .02*(65-age)/65 + diabetes*.2 + kidney*.25
        base_sens = max(.05, base_sens)
        if lisinopril_mg_now>0:
            peak_drop = max(0., avg_day_sbp - min(sbp_future))
            est = peak_drop / max(1., lisinopril_mg_now)
        else: est = base_sens + rng.normal(0,.05)
        sens_01 = float(np.clip(est/5.0 + rng.normal(0,.02), 0., 1.))

        row = {
            "avg_day_sbp":avg_day_sbp,"avg_day_dbp":avg_day_dbp,"baseline_hr":baseline_hr,
            "age":age,"sex":sex,"bmi":bmi,"diabetes":diabetes,"kidney":kidney,
            "sodium_mg_today":sodium_mg_today,"exercise_hours_today":exercise_hours_today,
            "lisinopril_mg_now":lisinopril_mg_now,"current_time_hour":current_time_hour,
            "sbp_now":sbp_now,"dbp_now":dbp_now,"hr_now":hr_now,SENS_COL:sens_01
        }
        for i,(s,d,h) in enumerate(zip(sbp_future,dbp_future,hr_future),1):
            row[f"sbp_{i}"]=s; row[f"dbp_{i}"]=d; row[f"hr_{i}"]=h
        rows.append(row)
    return pd.DataFrame(rows)

# ─────────────────── DATASET ─────────────────────────
class BPCSynthDataset(Dataset):
    def __init__(self, df:pd.DataFrame, scaler_X:StandardScaler=None,
                 scaler_y_seq:StandardScaler=None, fit_scalers=False):
        self.df = df.reset_index(drop=True)
        X_num = self.df[NUMERIC_FEATURES].values.astype(np.float32)
        sex  = self.df["sex"].map({'M':0.,'F':1.}).values.astype(np.float32).reshape(-1,1)
        self.X_raw = np.concatenate([X_num,sex],1)

        # Build (N, seq_len, 3) target tensor
        seq_parts=[]
        for i in range(SEQ_LEN):
            seq_parts.append(
                self.df[[SEQ_TARGET_SBPS[i], SEQ_TARGET_DBPS[i], SEQ_TARGET_HRS[i]]]
                    .values.astype(np.float32)
            )
        self.y_seq_raw = np.stack(seq_parts,1)  # (N, seq, 3)
        self.y_sens = self.df[SENS_COL].values.astype(np.float32).reshape(-1,1)

        # Fit scalers
        if fit_scalers:
            if scaler_X is None: scaler_X=StandardScaler().fit(self.X_raw)
            if scaler_y_seq is None:
                scaler_y_seq=StandardScaler().fit(self.y_seq_raw.reshape(len(self.y_seq_raw),-1))
        self.scaler_X, self.scaler_y_seq = scaler_X, scaler_y_seq

        self.X = self.scaler_X.transform(self.X_raw).astype(np.float32)
        flat = self.y_seq_raw.reshape(len(self.y_seq_raw),-1)
        self.y_seq = self.scaler_y_seq.transform(flat).reshape(len(self.y_seq_raw),SEQ_LEN,3)

    def __len__(self): return len(self.X)
    def __getitem__(self,idx): return self.X[idx], self.y_seq[idx], self.y_sens[idx]

# ──────────────────── MODEL ──────────────────────────
class Encoder(nn.Module):
    def __init__(self,in_dim,hid,lat):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,hid),nn.ReLU(),
                               nn.Linear(hid,lat),nn.ReLU())
    def forward(self,x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self,in_dim,hid,out_dim, n_layers=1):
        super().__init__()
        self.lstm=nn.LSTM(in_dim,hid,n_layers,batch_first=True)
        self.fc=nn.Linear(hid,out_dim)
    def forward(self,x,h): o,(hn,cn)=self.lstm(x,h); return self.fc(o),(hn,cn)

class Seq2SeqBP(nn.Module):
    def __init__(self,in_dim,lat,enc_hid,dec_in,dec_hid,out_dim,n_layers=1):
        super().__init__()
        self.encoder=Encoder(in_dim,enc_hid,lat)
        self.decoder=Decoder(dec_in,dec_hid,out_dim,n_layers)
        self.h0_proj=nn.Linear(lat,dec_hid); self.c0_proj=nn.Linear(lat,dec_hid)
        self.sens_head=nn.Sequential(
            nn.Linear(lat,max(8,lat//2)),nn.ReLU(),
            nn.Linear(max(8,lat//2),1), nn.Sigmoid())

    def forward(self,src,dec_init=None,trg_seq=None,teacher_forcing_ratio=.5):
        b = src.size(0)
        lat = self.encoder(src)
        sens = self.sens_head(lat)          # (b,1) in [0,1]

        h0=torch.tanh(self.h0_proj(lat)).unsqueeze(0)
        c0=torch.tanh(self.c0_proj(lat)).unsqueeze(0)
        hidden=(h0,c0)

        dec_in = dec_init if dec_init is not None else torch.zeros(b,1,3,device=src.device)
        outs=[]
        for t in range(SEQ_LEN):
            pred,hidden = self.decoder(dec_in,hidden)          # pred (b,1,3)
            outs.append(pred)
            dec_in = ( trg_seq[:,t].unsqueeze(1)
                       if trg_seq is not None and random.random()<teacher_forcing_ratio
                       else pred.detach() )
        return torch.cat(outs,1), sens                         # (b,seq,3), (b,1)

# ────────────── TRAIN / EVAL ─────────────────────────
def train_one_epoch(model,loader,opt,crit_seq,crit_sens,device):
    model.train(); tot=0
    sbp_i=NUMERIC_FEATURES.index("sbp_now"); dbp_i=NUMERIC_FEATURES.index("dbp_now")
    hr_i =NUMERIC_FEATURES.index("hr_now")
    for X,y_seq,y_sens in loader:
        X,y_seq,y_sens=X.to(device),y_seq.to(device),y_sens.to(device)
        dec_init=torch.cat([X[:,sbp_i:sbp_i+1],X[:,dbp_i:dbp_i+1],X[:,hr_i:hr_i+1]],1).unsqueeze(1)
        opt.zero_grad()
        p_seq,p_sens=model(X,dec_init,trg_seq=y_seq,teacher_forcing_ratio=TEACHER_FORCING_RATIO)
        loss=(crit_seq(p_seq,y_seq)+SENS_WEIGHT*crit_sens(p_sens,y_sens))
        loss.backward(); opt.step(); tot+=loss.item()*X.size(0)
    return tot/len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model,loader,crit_seq,crit_sens,device):
    model.eval(); tot=0
    sbp_i=NUMERIC_FEATURES.index("sbp_now"); dbp_i=NUMERIC_FEATURES.index("dbp_now")
    hr_i =NUMERIC_FEATURES.index("hr_now")
    for X,y_seq,y_sens in loader:
        X,y_seq,y_sens=X.to(device),y_seq.to(device),y_sens.to(device)
        dec_init=torch.cat([X[:,sbp_i:sbp_i+1],X[:,dbp_i:dbp_i+1],X[:,hr_i:hr_i+1]],1).unsqueeze(1)
        p_seq,p_sens=model(X,dec_init,trg_seq=None,teacher_forcing_ratio=0.)
        tot+=(crit_seq(p_seq,y_seq)+SENS_WEIGHT*crit_sens(p_sens,y_sens)).item()*X.size(0)
    return tot/len(loader.dataset)

# ────────────── TRAINING DRIVER ──────────────────────
def fit_on_synthetic(n_samples=50_000,epochs=EPOCHS,batch_size=BATCH_SIZE,lr=LR):
    print("Generating synthetic data...")
    df = generate_synthetic_dataframe(n_samples)
    idx=np.random.permutation(len(df)); split=int(.8*len(df))
    train_df, val_df = df.iloc[idx[:split]], df.iloc[idx[split:]]
    train_ds=BPCSynthDataset(train_df,fit_scalers=True)
    val_ds  =BPCSynthDataset(val_df,scaler_X=train_ds.scaler_X,
                             scaler_y_seq=train_ds.scaler_y_seq,fit_scalers=False)
    joblib.dump(train_ds.scaler_X,SCALER_X_PATH); joblib.dump(train_ds.scaler_y_seq,SCALER_Y_SEQ_PATH)
    print("Scalers saved.")

    tl=DataLoader(train_ds,batch_size,shuffle=True); vl=DataLoader(val_ds,batch_size)
    in_dim=train_ds.X.shape[1]
    model=Seq2SeqBP(in_dim,64,128,3,128,3).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr)
    c_seq,c_sens=nn.MSELoss(),nn.MSELoss()

    best=float("inf")
    for ep in range(1,epochs+1):
        tr=train_one_epoch(model,tl,opt,c_seq,c_sens,DEVICE)
        va=eval_one_epoch(model,vl,c_seq,c_sens,DEVICE)
        print(f"Epoch {ep:2d}/{epochs}  Train {tr:.5f}  Val {va:.5f}")
        if va<best:
            best=va
            torch.save({"model_state_dict":model.state_dict(),
                        "scaler_X_path":SCALER_X_PATH,
                        "scaler_y_seq_path":SCALER_Y_SEQ_PATH}, MODEL_OUT)
            print("  ↪ saved best model.")
    print("Done. Best val loss:",best)
    return MODEL_OUT,train_ds,model

# ──────────────── LOAD & PREDICT ─────────────────────
def load_model_and_scalers(path=MODEL_OUT):
    ckpt=torch.load(path,map_location=DEVICE)
    scaler_X=joblib.load(ckpt["scaler_X_path"]); scaler_y_seq=joblib.load(ckpt["scaler_y_seq_path"])
    model=Seq2SeqBP(len(scaler_X.mean_),64,128,3,128,3).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    return model,scaler_X,scaler_y_seq

@torch.no_grad()
def predict_from_dict(model,scaler_X,scaler_y_seq,row:Dict):
    sex_val={'M':0.,'F':1.}.get(row.get('sex','M'),0.)
    feats=np.array([row.get(k,0.) for k in NUMERIC_FEATURES]+[sex_val],dtype=np.float32).reshape(1,-1)
    X=torch.from_numpy(scaler_X.transform(feats)).to(DEVICE)

    sbp_i=NUMERIC_FEATURES.index("sbp_now"); dbp_i=NUMERIC_FEATURES.index("dbp_now")
    hr_i =NUMERIC_FEATURES.index("hr_now")
    dec_init=torch.cat([X[:,sbp_i:sbp_i+1],X[:,dbp_i:dbp_i+1],X[:,hr_i:hr_i+1]],1).unsqueeze(1)

    seq_scaled,sens=model(X,dec_init,trg_seq=None,teacher_forcing_ratio=0.0)
    seq_scaled=seq_scaled.squeeze(0).cpu().numpy()            # (seq,3)
    seq= scaler_y_seq.inverse_transform(seq_scaled.reshape(1,-1)).reshape(SEQ_LEN,3)
    sens_val=float(sens.item())

    base_hour=int(row.get("current_time_hour",0))
    out=[{"hour_offset":i+1,"target_hour":(base_hour+i+1)%24,
          "sbp_pred":float(seq[i,0]),"dbp_pred":float(seq[i,1]),"hr_pred":float(seq[i,2])}
         for i in range(SEQ_LEN)]
    return pd.DataFrame(out), sens_val

# ──────────────────── DEMO ──────────────────────────
if __name__=="__main__":
    mp,_,_=fit_on_synthetic()
    print("\nLoading saved model...")
    model,sc_X,sc_Y=load_model_and_scalers(mp)

    sample = {"avg_day_sbp":130,"avg_day_dbp":80,"baseline_hr":72,"age":79,"sex":"M",
              "bmi":28.5,"diabetes":1,"kidney":1,"sodium_mg_today":3200,"exercise_hours_today":0.0,
              "lisinopril_mg_now":10,"current_time_hour":14,"sbp_now":135,"dbp_now":84,"hr_now":78}
    df_p,sens=predict_from_dict(model,sc_X,sc_Y,sample)
    print("\nPredicted next 8 h vitals (SBP / DBP / HR):")
    print(df_p.to_string(index=False))
    print(f"\nPredicted sensitivity (0-1): {sens:.4f}")
    print("\nArtifacts saved:",SCALER_X_PATH,SCALER_Y_SEQ_PATH,MODEL_OUT)