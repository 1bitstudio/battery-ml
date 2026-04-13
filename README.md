# battery-ml

`battery-ml` — сервис для предсказания `SOH` (`State of Health`, состояния здоровья аккумулятора) по ранним циклам зарядки и разрядки. Проект работает как `Kafka`-воркер: получает данные из входного топика, выполняет инференс с помощью предобученной модели `PyTorch` и отправляет результат в выходной топик.

## Назначение проекта

Сервис:

- загружает предобученную модель `SOHTransformer` из `app/checkpoints_soh/...`
- подготавливает кривые зарядки и разрядки к фиксированному формату
- предсказывает `SOH` аккумулятора
- возвращает численное значение `SOH`, процент и целевой цикл
- работает асинхронно через `Kafka`

Основная точка входа: [app/ml_worker.py](/Users/a1/university/battery-ml/app/ml_worker.py)

## Структура проекта

```text
battery-ml/
├── app/
│   ├── ml_worker.py          # Kafka-воркер и пайплайн инференса
│   ├── models/               # архитектуры SOHLinear / SOHTransformer
│   ├── layers/               # слои трансформера и вспомогательные блоки
│   ├── checkpoints_soh/      # веса модели, scaler, args.json
│   └── utils/                # служебные модули
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Как работает сервис

Воркер:

1. подписывается на `Kafka`-топик `data`
2. валидирует входной JSON через `pydantic`
3. извлекает и ресемплирует кривые напряжения, тока и емкости
4. строит маску по наблюдаемым циклам
5. запускает инференс модели
6. публикует результат в топик `soh_responses`

Параметры `Kafka` по умолчанию в коде:

- `KAFKA_BOOTSTRAP`: `kafka:9092`
- `REQUEST_TOPIC`: `data`
- `RESPONSE_TOPIC`: `soh_responses`
- `GROUP_ID`: `battery-ml-worker`

## Формат входного сообщения

Сервис ожидает JSON с полями в `camelCase`:

```json
{
  "requestId": 1,
  "batteryInputData": {
    "nominalCapacityInAh": 2.5,
    "cycleData": [
      {
        "voltageInV": [3.7, 3.8, 3.9],
        "currentInA": [0.5, 0.5, -0.4],
        "chargeCapacityInAh": [0.1, 0.2, 0.2],
        "dischargeCapacityInAh": [0.0, 0.0, 0.1]
      }
    ],
    "obsCycles": 1,
    "SOCInterval": [0.0, 1.0]
  }
}
```

Пояснения:

- `obsCycles` — количество наблюдаемых циклов, используемых для предсказания
- `cycleData` — массив измерений по каждому циклу
- `SOCInterval` в текущей версии проходит валидацию, но напрямую в инференсе не используется

## Формат выходного сообщения

Успешный ответ:

```json
{
  "requestId": 1,
  "status": "ok",
  "result": {
    "predictedSoh": 0.94,
    "predictedSohPercent": 94.0,
    "targetCycle": 101
  }
}
```

Ответ при ошибке:

```json
{
  "requestId": 1,
  "status": "error",
  "error": "описание ошибки"
}
```

## Модель

Сейчас в проекте используется checkpoint:

`app/checkpoints_soh/SOH_SOHTransformer_dCALCE_ph100_dm64_df128_el2_dl1_lr0.0001_bs32_s2021`

Ключевые параметры из `args.json`:

- модель: `SOHTransformer`
- датасет: `CALCE`
- `early_cycle_threshold`: `100`
- `charge_discharge_length`: `100`
- `pred_horizon`: `100`

Это означает, что модель использует до 100 ранних циклов и предсказывает `SOH` на 100 циклов вперед относительно точки наблюдения.

## Требования

- `Docker` и `Docker Compose`
- доступный `Kafka`-брокер по адресу `kafka:9092`
- внешняя `Docker`-сеть `code_battery_network`

Python-зависимости перечислены в [requirements.txt](/Users/a1/university/battery-ml/requirements.txt):

- `torch`
- `numpy`
- `aiokafka`
- `safetensors`
- `joblib`
- `scikit_learn`
- `pydantic`

## Запуск через Docker Compose

Если внешняя сеть еще не создана:

```bash
docker network create code_battery_network
```

Сборка и запуск:

```bash
docker compose up --build -d
```

Остановка:

```bash
docker compose down
```

Просмотр логов:

```bash
docker compose logs -f app
```

## Локальный запуск без Docker

Создание виртуального окружения и установка зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Запуск воркера:

```bash
python app/ml_worker.py
```

## Важные замечания по реализации

- сервис является `Kafka`-воркером, а не HTTP API
- порт `8000` указан в `docker-compose.yml`, но `ml_worker.py` не поднимает веб-сервер
- в репозитории также есть [app/app_a.py](/Users/a1/university/battery-ml/app/app_a.py) — это отдельный прототип `FastAPI` для HTTP-инференса, который по умолчанию в `Docker` не используется

