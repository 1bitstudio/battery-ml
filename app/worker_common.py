from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import joblib
import numpy as np
import torch
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


def to_camel(value: str) -> str:
    special = {
        "a": "A",
        "ah": "Ah",
        "c": "C",
        "ohm": "Ohm",
        "s": "S",
        "soc": "SOC",
        "v": "V",
    }

    parts = value.split("_")
    result = parts[0]
    for part in parts[1:]:
        result += special.get(part, part.capitalize())
    return result


class CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",
    )


class Cycle(CamelModel):
    voltage_in_v: list[float]
    current_in_a: list[float]
    charge_capacity_in_ah: list[float]
    discharge_capacity_in_ah: list[float]
    time_in_s: list[float] | None = None
    temperature_in_c: list[float] | None = None
    internal_resistance_in_ohm: list[float] | None = None


class BatteryData(CamelModel):
    nominal_capacity_in_ah: float
    cycle_data: list[Cycle]
    obs_cycles: int
    form_factor: str | None = None
    anode_composition: str | None = None
    cathode_composition: str | None = None
    electrolyte_composition: str | None = None
    soc_interval: list[float] = Field(
        default_factory=lambda: [0.0, 1.0],
        alias="SOCInterval",
    )


class SOHRequest(CamelModel):
    request_id: int
    cycle_numbers: list[int] | None = None
    battery_input_data: BatteryData


class RULRequest(CamelModel):
    request_id: int
    battery_input_data: BatteryData


@dataclass(frozen=True)
class WorkerConfig:
    service_name: str
    kafka_bootstrap: str
    request_topic: str
    response_topic: str
    group_id: str
    checkpoint_path: str


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    scaler: Any | None
    args: Any
    checkpoint_path: str


def build_worker_config(
    service_name: str,
    default_request_topic: str,
    default_response_topic: str,
    default_group_id: str,
    default_checkpoint_path: str,
) -> WorkerConfig:
    return WorkerConfig(
        service_name=service_name,
        kafka_bootstrap=os.getenv("KAFKA_BOOTSTRAP", "kafka:9092"),
        request_topic=os.getenv("REQUEST_TOPIC", default_request_topic),
        response_topic=os.getenv("RESPONSE_TOPIC", default_response_topic),
        group_id=os.getenv("GROUP_ID", default_group_id),
        checkpoint_path=os.getenv("CHECKPOINT_PATH", default_checkpoint_path),
    )


def resolve_checkpoint_path(base_path: str) -> str:
    normalized = os.path.abspath(base_path)
    args_path = os.path.join(normalized, "args.json")
    if os.path.isfile(args_path):
        return normalized

    if not os.path.isdir(normalized):
        raise FileNotFoundError(f"Checkpoint path does not exist: {normalized}")

    candidates: list[str] = []
    for entry in sorted(os.listdir(normalized)):
        candidate = os.path.join(normalized, entry)
        if os.path.isfile(os.path.join(candidate, "args.json")):
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find args.json in checkpoint path: {normalized}"
        )
    raise ValueError(
        "Multiple checkpoints found. Set CHECKPOINT_PATH to a concrete directory: "
        + ", ".join(candidates)
    )


def load_model_class(model_name: str):
    module_name = f"models.{model_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unsupported model '{model_name}'") from exc

    if not hasattr(module, "Model"):
        raise ValueError(f"Model module '{module_name}' does not expose Model")
    return module.Model


def load_scaler(checkpoint_path: str):
    scaler_path = os.path.join(checkpoint_path, "label_scaler")
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)


def restore_prediction(prediction: float, scaler: Any | None) -> float:
    if scaler is None:
        return float(prediction)

    value = np.array([[prediction]], dtype=np.float32)

    if hasattr(scaler, "inverse_transform"):
        restored = scaler.inverse_transform(value)
        return float(np.squeeze(restored))

    if hasattr(scaler, "var_") and hasattr(scaler, "mean_"):
        std = np.sqrt(scaler.var_[0])
        mean = scaler.mean_[0]
        return float(prediction * std + mean)

    return float(prediction)


@lru_cache(maxsize=None)
def load_model_bundle(base_path: str) -> ModelBundle:
    checkpoint_path = resolve_checkpoint_path(base_path)

    with open(os.path.join(checkpoint_path, "args.json")) as file:
        args_dict = json.load(file)

    args = types.SimpleNamespace(**args_dict)
    model_class = load_model_class(args.model)
    model = model_class(args)

    weights_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(weights_path):
        state_dict = load_file(weights_path)
    else:
        state_dict = torch.load(
            os.path.join(checkpoint_path, "checkpoint.pth"),
            map_location="cpu",
        )

    model.load_state_dict(state_dict)
    model.eval()

    return ModelBundle(
        model=model,
        scaler=load_scaler(checkpoint_path),
        args=args,
        checkpoint_path=checkpoint_path,
    )


def resample(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(target_len)

    bases = np.arange(1, len(values) + 1)
    points = np.linspace(1, len(values) + 1, num=target_len)
    return np.interp(points, bases, values)


def extract_curves(data: dict[str, Any], cycle_limit: int, curve_len: int) -> np.ndarray:
    half = curve_len // 2
    nominal_capacity = data["nominal_capacity_in_ah"]
    curves = []

    for index in range(cycle_limit):
        if index >= len(data["cycle_data"]):
            curves.append(np.zeros((1, 3, curve_len)))
            continue

        cycle = data["cycle_data"][index]
        voltage = np.array(cycle["voltage_in_v"])
        current = np.array(cycle["current_in_a"])
        current_c = current / nominal_capacity
        charge_capacity = np.array(cycle["charge_capacity_in_ah"])
        discharge_capacity = np.array(cycle["discharge_capacity_in_ah"])

        try:
            charge_end = np.nonzero(current_c >= 0.01)[0][-1]
        except IndexError:
            curves.append(np.zeros((1, 3, curve_len)))
            continue

        discharge_voltage = voltage[charge_end:]
        discharge_capacity = discharge_capacity[charge_end:]
        discharge_current = current[charge_end:]
        discharge_current_c = discharge_current / nominal_capacity

        mask = np.abs(discharge_current_c) > 0.01
        discharge_voltage = discharge_voltage[mask]
        discharge_capacity = discharge_capacity[mask]
        discharge_current = discharge_current[mask]

        charge_voltage = voltage[:charge_end]
        charge_current = current[:charge_end]
        charge_capacity = charge_capacity[:charge_end]

        discharge_voltage = resample(discharge_voltage, half)
        discharge_current = resample(discharge_current, half)
        discharge_capacity = resample(discharge_capacity, half)
        charge_voltage = resample(charge_voltage, half)
        charge_current = resample(charge_current, half)
        charge_capacity = resample(charge_capacity, half)

        voltage_curve = np.concatenate([charge_voltage, discharge_voltage])
        current_curve = np.concatenate([charge_current, discharge_current])
        capacity_curve = np.concatenate([charge_capacity, discharge_capacity])

        voltage_curve = (
            voltage_curve / max(np.abs(voltage_curve).max(), 1e-8)
        ).reshape(1, curve_len)
        current_curve = (current_curve / nominal_capacity).reshape(1, curve_len)
        capacity_curve = (capacity_curve / nominal_capacity).reshape(1, curve_len)

        curve = np.concatenate([voltage_curve, current_curve, capacity_curve], axis=0)
        curves.append(curve.reshape(1, 3, curve_len))

    return np.concatenate(curves, axis=0)


def validate_battery_data(battery_data: BatteryData, args: Any) -> None:
    if battery_data.obs_cycles < 1:
        raise ValueError("obsCycles must be greater than zero")
    if battery_data.obs_cycles > len(battery_data.cycle_data):
        raise ValueError("obsCycles exceeds cycleData length")
    if battery_data.obs_cycles > args.early_cycle_threshold:
        raise ValueError(
            "obsCycles exceeds model early_cycle_threshold "
            f"({args.early_cycle_threshold})"
        )


def predict_scalar(bundle: ModelBundle, battery_data: BatteryData) -> float:
    validate_battery_data(battery_data, bundle.args)

    curves = extract_curves(
        battery_data.model_dump(),
        bundle.args.early_cycle_threshold,
        bundle.args.charge_discharge_length,
    )

    mask = np.zeros(bundle.args.early_cycle_threshold, dtype=np.float32)
    mask[:battery_data.obs_cycles] = 1.0

    curves_tensor = torch.tensor(curves, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    expanded_mask = mask_tensor.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(curves_tensor)
    curves_tensor[expanded_mask == 0] = 0

    with torch.no_grad():
        output = bundle.model(curves_tensor, mask_tensor)

    return restore_prediction(output.squeeze().item(), bundle.scaler)


async def run_worker(
    config: WorkerConfig,
    request_model: type[BaseModel],
    success_handler: Callable[[BaseModel, ModelBundle], dict[str, Any]],
    error_handler: Callable[[Any, Exception], dict[str, Any]],
) -> None:
    logging.basicConfig(level=logging.INFO)
    bundle = load_model_bundle(config.checkpoint_path)

    consumer = AIOKafkaConsumer(
        config.request_topic,
        bootstrap_servers=config.kafka_bootstrap,
        group_id=config.group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=config.kafka_bootstrap,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    await consumer.start()
    await producer.start()

    logger.info(
        "%s worker started. request_topic=%s response_topic=%s checkpoint=%s",
        config.service_name,
        config.request_topic,
        config.response_topic,
        bundle.checkpoint_path,
    )

    try:
        async for message in consumer:
            raw = message.value
            request_id = raw.get("requestId") if isinstance(raw, dict) else None

            try:
                request = request_model.model_validate(raw)
                request_id = getattr(request, "request_id", request_id)
                response = success_handler(request, bundle)
                logger.info("%s requestId=%s processed", config.service_name, request_id)
            except Exception as exc:
                response = error_handler(request_id, exc)
                logger.exception("%s requestId=%s failed", config.service_name, request_id)

            await producer.send_and_wait(config.response_topic, response)
    finally:
        await consumer.stop()
        await producer.stop()


def run_worker_sync(
    config: WorkerConfig,
    request_model: type[BaseModel],
    success_handler: Callable[[BaseModel, ModelBundle], dict[str, Any]],
    error_handler: Callable[[Any, Exception], dict[str, Any]],
) -> None:
    asyncio.run(run_worker(config, request_model, success_handler, error_handler))
