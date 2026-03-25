FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
    apt-get install -y stockfish && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pretraining/ pretraining/
COPY rl/ rl/

CMD ["python", "rl/train_hf.py"]
