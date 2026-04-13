import asyncio
import json
import os
import types
import uuid

import numpy as np
import torch
import joblib

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from safetensors.torch import load_file
import logging


from models import SOHLinear, SOHTransformer


KAFKA_BOOTSTRAP = "kafka:9092"
REQUEST_TOPIC = "data"
RESPONSE_TOPIC = "soh_responses"
GROUP_ID = "battery-ml-worker"

CKPT_PATH = "./app/checkpoints_soh/SOH_SOHTransformer_dCALCE_ph100_dm64_df128_el2_dl1_lr0.0001_bs32_s2021"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from pydantic import BaseModel, ConfigDict, Field

def to_camel(s: str) -> str:
    special = {
        "ah": "Ah",
        "v": "V",
        "a": "A",
        "soc": "SOC"
    }

    parts = s.split("_")
    result = parts[0]

    for p in parts[1:]:
        if p in special:
            result += special[p]
        else:
            result += p.capitalize()

    return result


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )

from typing import List

class Cycle(CamelModel):
    voltage_in_V: List[float]
    current_in_A: List[float]
    charge_capacity_in_Ah: List[float]
    discharge_capacity_in_Ah: List[float]


class BatteryData(CamelModel):
    nominal_capacity_in_Ah: float
    cycle_data: List[Cycle]
    obs_cycles: int
    soc_interval: List[float] = Field(alias="SOCInterval")


class RequestModel(CamelModel):
    request_id: int
    battery_input_data: BatteryData



# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# PREPROCESS
# ---------------------------------------------------------------
def resample(arr, target_len):
    if len(arr) < 2:
        return np.zeros(target_len)
    bases = np.arange(1, len(arr) + 1)
    pts = np.linspace(1, len(arr) + 1, num=target_len)
    return np.interp(pts, bases, arr)


def extract_curves(data, L, curve_len):
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
            except:
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


# ---------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------
def predict(data, obs_cycles):
    curves = extract_curves(
        data,
        args.early_cycle_threshold,
        args.charge_discharge_length,
    )

    mask = np.zeros(args.early_cycle_threshold, dtype=np.float32)
    mask[:obs_cycles] = 1.0

    curves_t = torch.tensor(curves, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    tmp = mask_t.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(curves_t)
    curves_t[tmp == 0] = 0

    with torch.no_grad():
        output = model(curves_t, mask_t)

    pred_scaled = output.squeeze().item()

    std = np.sqrt(scaler.var_[0])
    mean = scaler.mean_[0]

    pred_soh = pred_scaled * std + mean

    return {
        "predicted_soh": float(pred_soh),
        "predicted_soh_percent": float(pred_soh * 100),
        "target_cycle": int(obs_cycles + args.pred_horizon),
    }


# ---------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------
async def main():
    consumer = AIOKafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )

    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    await consumer.start()
    await producer.start()

    logger.info("🚀 ML worker started...")

    try:
        async for msg in consumer:
            raw = msg.value
            request = RequestModel(**raw)

            request_id = request.request_id

            try:
                data_dict = request.battery_input_data.model_dump()
                result = predict(
                    data_dict,
                    request.obs_cycles
                )

                response = {
                    "request_id": request_id,
                    "status": "ok",
                    "result": result
                }
                


            except Exception as e:
                response = {
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e)
                }

            response = RequestModel(**response).model_dump(by_alias=True)
            await producer.send_and_wait(RESPONSE_TOPIC, response)

    finally:
        await consumer.stop()
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())