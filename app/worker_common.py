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
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
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
    anode_composition: str | None = Field(
        default=None,
        validation_alias=AliasChoices("anodeComposition", "anodeMaterial"),
    )
    cathode_composition: str | None = Field(
        default=None,
        validation_alias=AliasChoices("cathodeComposition", "cathodeMaterial"),
    )
    electrolyte_composition: str | None = Field(
        default=None,
        validation_alias=AliasChoices("electrolyteComposition", "electrolyteMaterial"),
    )
    charge_protocol: str | None = None
    discharge_protocol: str | None = None
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


def payload_to_log(payload: Any, limit: int = 2000) -> str:
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        text = str(payload)

    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def decode_kafka_message(raw_value: bytes) -> dict[str, Any] | None:
    if raw_value is None:
        logger.warning("Kafka message skipped: empty value")
        return None

    text = raw_value.decode("utf-8", errors="replace").strip()
    if not text:
        logger.warning("Kafka message skipped: blank value")
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Kafka message skipped: invalid JSON: %s", text[:300])
        return None

    if not isinstance(payload, dict):
        logger.warning("Kafka message skipped: JSON root is not an object")
        return None

    return payload


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


def safe_array(values: list[float] | None) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def summarize_series(values: list[float] | None) -> list[float]:
    array = safe_array(values)
    if array.size == 0:
        return [0.0] * 9

    first = float(array[0])
    last = float(array[-1])
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    mean = float(np.mean(array))
    std = float(np.std(array))
    delta = last - first
    spread = maximum - minimum
    sample_count = float(array.size)

    return [mean, std, minimum, maximum, first, last, delta, spread, sample_count]


def build_temperature_features(battery_data: BatteryData, args: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not getattr(args, "use_temperature", False):
        return None, None

    summary_dim = int(getattr(args, "temperature_summary_dim", 18))
    features = []
    mask = []

    for index in range(args.early_cycle_threshold):
        if index >= len(battery_data.cycle_data):
            features.append([0.0] * summary_dim)
            mask.append(0.0)
            continue

        cycle = battery_data.cycle_data[index]
        vector = summarize_series(cycle.temperature_in_c) + summarize_series(
            cycle.internal_resistance_in_ohm
        )
        if len(vector) < summary_dim:
            vector.extend([0.0] * (summary_dim - len(vector)))
        else:
            vector = vector[:summary_dim]

        has_temperature = bool(cycle.temperature_in_c) or bool(cycle.internal_resistance_in_ohm)
        features.append(vector)
        mask.append(1.0 if has_temperature else 0.0)

    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    return feature_tensor, mask_tensor


def get_soc_bounds(battery_data: BatteryData) -> tuple[float, float]:
    if len(battery_data.soc_interval) >= 2:
        start = float(battery_data.soc_interval[0])
        end = float(battery_data.soc_interval[1])
        return start, end
    return 0.0, 1.0


def get_voltage_limits(battery_data: BatteryData) -> tuple[float, float]:
    observed_cycles = battery_data.cycle_data[:battery_data.obs_cycles]
    values: list[float] = []
    for cycle in observed_cycles:
        values.extend(cycle.voltage_in_v)

    if not values:
        return 0.0, 0.0

    return float(min(values)), float(max(values))


def get_numeric_feature_value(battery_data: BatteryData, key: str) -> float:
    soc_start, soc_end = get_soc_bounds(battery_data)
    min_voltage, max_voltage = get_voltage_limits(battery_data)
    soc_span = soc_end - soc_start

    numeric_map = {
        "nominal_capacity_in_Ah": float(battery_data.nominal_capacity_in_ah),
        "depth_of_charge": float(max(soc_span, 0.0)),
        "depth_of_discharge": float(max(soc_span, 0.0)),
        "max_voltage_limit_in_V": float(max_voltage),
        "min_voltage_limit_in_V": float(min_voltage),
        "soc_start": float(soc_start),
        "soc_end": float(soc_end),
        "soc_interval": float(soc_span),
    }
    return numeric_map.get(key, 0.0)


def normalize_form_factor(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    mapping = {
        "18650": "cylindrical_18650",
        "cylindrical": "cylindrical_18650",
        "cylindrical_18650": "cylindrical_18650",
        "prismatic": "prismatic",
        "pouch": "pouch",
        "coin": "coin",
    }
    return mapping.get(normalized, value)


def get_categorical_feature_value(battery_data: BatteryData, key: str) -> str | None:
    categorical_map = {
        "form_factor": normalize_form_factor(battery_data.form_factor),
        "anode_material": battery_data.anode_composition,
        "cathode_material": battery_data.cathode_composition,
        "electrolyte_material": battery_data.electrolyte_composition,
        "charge_protocol": battery_data.charge_protocol,
        "discharge_protocol": battery_data.discharge_protocol,
    }
    return categorical_map.get(key)


def encode_categorical_value(args: Any, key: str, raw_value: str | None) -> int:
    value_to_id = getattr(args, "metadata_value_to_id", {}).get(key, {})
    if raw_value is None:
        return int(value_to_id.get("__MISSING__", 1))
    if raw_value in value_to_id:
        return int(value_to_id[raw_value])
    return int(value_to_id.get("__UNK__", 0))


def build_metadata_tensors(battery_data: BatteryData, args: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not getattr(args, "use_metadata", False):
        return None, None

    numeric_keys = list(getattr(args, "metadata_numeric_keys", []))
    categorical_keys = list(getattr(args, "metadata_categorical_keys", []))

    numeric_tensor = None
    categorical_tensor = None

    if numeric_keys:
        numeric_values = [get_numeric_feature_value(battery_data, key) for key in numeric_keys]
        numeric_tensor = torch.tensor([numeric_values], dtype=torch.float32)

    if categorical_keys:
        categorical_values = [
            encode_categorical_value(
                args,
                key,
                get_categorical_feature_value(battery_data, key),
            )
            for key in categorical_keys
        ]
        categorical_tensor = torch.tensor([categorical_values], dtype=torch.long)

    return numeric_tensor, categorical_tensor


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
    temperature_features, temperature_mask = build_temperature_features(
        battery_data, bundle.args
    )
    static_numeric, static_categorical = build_metadata_tensors(
        battery_data, bundle.args
    )

    expanded_mask = mask_tensor.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(curves_tensor)
    curves_tensor[expanded_mask == 0] = 0

    with torch.no_grad():
        output = bundle.model(
            curves_tensor,
            mask_tensor,
            temperature_features=temperature_features,
            temperature_mask=temperature_mask,
            static_numeric=static_numeric,
            static_categorical=static_categorical,
        )

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
            raw = decode_kafka_message(message.value)
            if raw is None:
                logger.warning(
                    "%s skipped message from topic=%s partition=%s offset=%s",
                    config.service_name,
                    message.topic,
                    message.partition,
                    message.offset,
                )
                continue
            request_id = raw.get("requestId") if isinstance(raw, dict) else None
            logger.info(
                "%s received message from topic=%s partition=%s offset=%s requestId=%s payload=%s",
                config.service_name,
                message.topic,
                message.partition,
                message.offset,
                request_id,
                payload_to_log(raw),
            )

            try:
                request = request_model.model_validate(raw)
                request_id = getattr(request, "request_id", request_id)
                response = success_handler(request, bundle)
                logger.info("%s requestId=%s processed", config.service_name, request_id)
            except Exception as exc:
                response = error_handler(request_id, exc)
                logger.exception("%s requestId=%s failed", config.service_name, request_id)

            logger.info(
                "%s sending message to topic=%s requestId=%s payload=%s",
                config.service_name,
                config.response_topic,
                request_id,
                payload_to_log(response),
            )
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
