# Slim Python image for fast builds (CPU only)
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code
COPY . .

# Run handler
CMD ["python3", "-u", "handler.py"]
