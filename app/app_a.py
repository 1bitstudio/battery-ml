import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from worker_common import BatteryData, load_model_bundle, predict_scalar


app = FastAPI(title="SOH Prediction API")

DEFAULT_CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "./app/checkpoints_soh")
bundle = load_model_bundle(DEFAULT_CHECKPOINT_PATH)


class PredictRequest(BaseModel):
    data: BatteryData
    file_name: str = "custom"


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        prediction = predict_scalar(bundle, request.data)
        target_cycle = request.data.obs_cycles + bundle.args.pred_horizon

        return {
            "predicted_soh": float(prediction),
            "target_cycle": int(target_cycle),
            "model_name": bundle.args.model,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
