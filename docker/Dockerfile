FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal system deps; libgomp1 is needed at runtime for xgboost openmp
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Use the correct port your service listens on (8080 in your logs)
EXPOSE 8080

# Start the service (adjust path if your app lives in src/)
CMD ["python", "src/app.py"]