from worker_common import (
    ModelBundle,
    RULRequest,
    build_worker_config,
    predict_scalar,
    run_worker_sync,
)


DEFAULT_CHECKPOINT_PATH = "./app/checkpoints_rul"


def build_success_response(request: RULRequest, bundle: ModelBundle) -> dict:
    prediction = predict_scalar(bundle, request.battery_input_data)

    return {
        "requestId": request.request_id,
        "status": "ok",
        "predictionRul": float(prediction),
        "error": None,
    }


def build_error_response(request_id, error: Exception) -> dict:
    return {
        "requestId": request_id,
        "status": "error",
        "predictionRul": None,
        "error": str(error),
    }


def main() -> None:
    config = build_worker_config(
        service_name="rul",
        default_request_topic="rul-data",
        default_response_topic="rul_responses",
        default_group_id="battery-ml-rul-worker",
        default_checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    )
    run_worker_sync(config, RULRequest, build_success_response, build_error_response)


if __name__ == "__main__":
    main()
