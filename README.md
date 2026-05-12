# battery-ml

`battery-ml` — сервисы для предсказания `SOH` (состояния здоровья аккумулятора) и `RUL` (оставшегося ресурса по циклам). Проект работает как `Kafka`-воркеры: получает данные из входных топиков, выполняет инференс с помощью предобученных моделей `PyTorch` и отправляет результат в выходные топики.

## Назначение проекта

Сервисы:

- загружают предобученные модели из `app/checkpoints_soh/...` и `app/checkpoints_rul/...`
- для `SOH` используют архитектуру `SOHTransformerV4`
- подготавливают кривые зарядки и разрядки к фиксированному формату
- предсказывают `SOH` аккумулятора и `RUL`
- работают асинхронно через `Kafka`

Основная точка входа: [app/ml_worker.py](/Users/a1/university/battery-ml/app/ml_worker.py)

## Структура проекта

```text
battery-ml/
├── app/
│   ├── ml_worker.py          # совместимая точка входа для SOH-воркера
│   ├── ml_worker_soh.py      # Kafka-воркер для SOH
│   ├── ml_worker_rul.py      # Kafka-воркер для RUL
│   ├── worker_common.py      # общая загрузка модели и общий цикл Kafka
│   ├── models/               # архитектуры моделей, включая SOHTransformerV4
│   ├── layers/               # слои трансформера и вспомогательные блоки
│   ├── checkpoints_soh/      # веса модели SOH
│   ├── checkpoints_rul/      # веса модели RUL
│   └── utils/                # служебные модули
├── Dockerfile.soh
├── Dockerfile.rul
├── docker-compose.soh.yml
├── docker-compose.rul.yml
└── requirements.txt
```

## Как работает сервис

Для `SOH` воркер:

1. подписывается на `Kafka`-топик `data`
2. валидирует входной JSON через `pydantic`
3. извлекает и ресемплирует кривые напряжения, тока и емкости
4. строит маску по наблюдаемым циклам
5. запускает инференс модели `SOHTransformerV4`
6. публикует результат в топик `soh_responses`

Параметры `Kafka` по умолчанию в коде:

- `KAFKA_BOOTSTRAP`: `kafka:9092`
- `REQUEST_TOPIC`: `data`
- `RESPONSE_TOPIC`: `soh_responses`
- `GROUP_ID`: `battery-ml-soh-worker`

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
- дополнительные поля вроде температуры, внутреннего сопротивления и состава аккумулятора принимаются и не ломают обработку

## Формат выходного сообщения

Успешный ответ для `SOH`:

```json
{
  "requestId": 1,
  "status": "ok",
  "result": {
    "predictedSoh": 0.94,
    "targetCycle": 101
  },
  "error": null
}
```

Успешный ответ для `RUL`:

```json
{
  "requestId": 1,
  "status": "ok",
  "predictionRul": 450.0,
  "error": null
}
```

Ответ при ошибке:

```json
{
  "requestId": 1,
  "status": "error",
  "result": null,
  "error": "описание ошибки"
}
```

## Модель

Для `SOH` по умолчанию используется каталог:

`app/checkpoints_soh`

При запуске воркер ищет внутри него конкретный чекпоинт и загружает `args.json`, веса и `label_scaler`.

Ожидается, что в `args.json` для `SOH` будет указана модель:

- модель: `SOHTransformerV4`
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

Запуск `SOH`:

```bash
docker compose -f docker-compose.soh.yml up --build -d
```

Запуск `RUL`:

```bash
docker compose -f docker-compose.rul.yml up --build -d
```

Остановка:

```bash
docker compose -f docker-compose.soh.yml down
docker compose -f docker-compose.rul.yml down
```

Просмотр логов:

```bash
docker compose -f docker-compose.soh.yml logs -f battery-ml-soh
docker compose -f docker-compose.rul.yml logs -f battery-ml-rul
```

## Локальный запуск без Docker

Создание виртуального окружения и установка зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Запуск `SOH`-воркера:

```bash
python app/ml_worker.py
```

## Важные замечания по реализации

- основные сервисы являются `Kafka`-воркерами, а не `HTTP`-сервисами
- `SOH` и `RUL` запускаются отдельными контейнерами и могут работать одновременно
- [app/app_a.py](/Users/a1/university/battery-ml/app/app_a.py) — это отдельный прототип `FastAPI` для `SOH`, он тоже переведён на общую загрузку модели и будет поднимать ту модель, которая лежит в `app/checkpoints_soh`
