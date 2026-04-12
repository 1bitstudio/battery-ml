FROM python:3.10-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --timeout 120 \
  --extra-index-url https://download.pytorch.org/whl/cpu
COPY ./app /code/app

CMD ["python", "app/ml_worker.py"]