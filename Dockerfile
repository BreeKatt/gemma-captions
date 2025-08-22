# Slim Python image for fast builds (CPU only)
FROM python:3.10-slim

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["python3", "-u", "handler.py"]
