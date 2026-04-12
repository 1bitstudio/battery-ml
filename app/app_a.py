import os
import json
import types

import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from safetensors.torch import load_file
from app.models import SOHLinear
from app.models import SOHTransformer

app = FastAPI(title="SOH Prediction API")

class Cycle(BaseModel):
    voltage_in_V: List[float]
    current_in_A: List[float]
    charge_capacity_in_Ah: List[float]
    discharge_capacity_in_Ah: List[float]


class BatteryData(BaseModel):
    nominal_capacity_in_Ah: float
    cycle_data: List[Cycle]
    SOC_interval: List[float] = Field(default=[0, 1])


class PredictRequest(BaseModel):
    data: BatteryData
    file_name: str = "custom"
    obs_cycles: int


def resample(arr, target_len):
    if len(arr) < 2:
        return np.zeros(target_len)
    bases = np.arange(1, len(arr) + 1)
    pts = np.linspace(1, len(arr) + 1, num=target_len, endpoint=True)
    return np.interp(pts, bases, arr)


def extract_curves(data, file_name, L, curve_len):
    half = curve_len // 2
    nom = data["nominal_capacity_in_Ah"]

    curves = []

    for idx in range(L):
        if idx < len(data["cycle_data"]):
            cyc = data["cycle_data"][idx]

            voltage = np.array(cyc["voltage_in_V"])
            current = np.array(cyc["current_in_A"])
            current_c = current / nom

            charge_cap = np.array(cyc["charge_capacity_in_Ah"])
            discharge_cap = np.array(cyc["discharge_capacity_in_Ah"])

            try:
                charge_end = np.nonzero(current_c >= 0.01)[0][-1]
            except IndexError:
                curves.append(np.zeros((1, 3, curve_len)))
                continue

            dv = voltage[charge_end:]
            dc = discharge_cap[charge_end:]
            di = current[charge_end:]
            di_c = di / nom

            mask = np.abs(di_c) > 0.01
            dv, dc, di = dv[mask], dc[mask], di[mask]

            cv = voltage[:charge_end]
            cc = charge_cap[:charge_end]
            ci = current[:charge_end]

            dv = resample(dv, half)
            di = resample(di, half)
            dc = resample(dc, half)
            cv = resample(cv, half)
            ci = resample(ci, half)
            cc = resample(cc, half)

            v = np.concatenate([cv, dv])
            c = np.concatenate([ci, di])
            cap = np.concatenate([cc, dc])

            v = (v / max(np.abs(v).max(), 1e-8)).reshape(1, curve_len)
            c = (c / nom).reshape(1, curve_len)
            cap = (cap / nom).reshape(1, curve_len)

            curve = np.concatenate([v, c, cap], axis=0)
        else:
            curve = np.zeros((3, curve_len))

        curves.append(curve.reshape(1, 3, curve_len))

    return np.concatenate(curves, axis=0)


CKPT_PATH = "./app/checkpoints_soh/SOH_SOHTransformer_dCALCE_ph100_dm64_df128_el2_dl1_lr0.0001_bs32_s2021"

def load_model():
    with open(os.path.join(CKPT_PATH, "args.json")) as f:
        args_dict = json.load(f)

    args = types.SimpleNamespace(**args_dict)

    if args.model == "SOHLinear":
        model = SOHLinear.Model(args)
    else:
        model = SOHTransformer.Model(args)

    weights_path = os.path.join(CKPT_PATH, "model.safetensors")

    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(
            os.path.join(CKPT_PATH, "checkpoint.pth"),
            map_location="cpu"
        )

    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(os.path.join(CKPT_PATH, "label_scaler"))

    return model, scaler, args


model, scaler, args = load_model()


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        data = req.data.dict()

        # --- validation ---
        if req.obs_cycles > len(data["cycle_data"]):
            raise ValueError("obs_cycles > number of cycles")

        curves = extract_curves(
            data,
            req.file_name,
            args.early_cycle_threshold,
            args.charge_discharge_length,
        )

        mask = np.zeros(args.early_cycle_threshold, dtype=np.float32)
        mask[:req.obs_cycles] = 1.0

        curves_t = torch.tensor(curves, dtype=torch.float32).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        tmp_mask = mask_t.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(curves_t)
        curves_t[tmp_mask == 0] = 0

        with torch.no_grad():
            output = model(curves_t, mask_t)

        pred_scaled = output.squeeze().item()

        std = np.sqrt(scaler.var_[0])
        mean = scaler.mean_[0]

        pred_soh = pred_scaled * std + mean

        target_cycle = req.obs_cycles + args.pred_horizon

        return {
            "predicted_soh": float(pred_soh),
            "predicted_soh_percent": float(pred_soh * 100),
            "target_cycle": int(target_cycle)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))