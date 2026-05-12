from worker_common import (
    ModelBundle,
    SOHRequest,
    build_worker_config,
    predict_scalar,
    run_worker_sync,
)


DEFAULT_CHECKPOINT_PATH = "./app/checkpoints_soh"


def build_success_response(request: SOHRequest, bundle: ModelBundle) -> dict:
    battery_data = request.battery_input_data
    if request.cycle_numbers is not None and len(request.cycle_numbers) < battery_data.obs_cycles:
        raise ValueError("cycleNumbers length is smaller than obsCycles")

    prediction = predict_scalar(bundle, battery_data)
    target_cycle = battery_data.obs_cycles + bundle.args.pred_horizon
    if request.cycle_numbers:
        target_cycle = request.cycle_numbers[battery_data.obs_cycles - 1] + bundle.args.pred_horizon

    return {
        "requestId": request.request_id,
        "status": "ok",
        "result": {
            "predictedSoh": float(prediction),
            "targetCycle": int(target_cycle),
        },
        "error": None,
    }


def build_error_response(request_id, error: Exception) -> dict:
    return {
        "requestId": request_id,
        "status": "error",
        "result": None,
        "error": str(error),
    }


def main() -> None:
    config = build_worker_config(
        service_name="soh",
        default_request_topic="soh-data",
        default_response_topic="soh_responses",
        default_group_id="battery-ml-soh-worker",
        default_checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    )
    run_worker_sync(config, SOHRequest, build_success_response, build_error_response)


if __name__ == "__main__":
    main()
