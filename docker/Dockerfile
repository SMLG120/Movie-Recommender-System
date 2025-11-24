FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for scikit-surprise, implicit, xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libgomp1 \
      gcc \
      g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PDM globally
RUN pip install --no-cache-dir pdm

# Copy PDM metadata first for layer caching
COPY pyproject.toml pdm.lock ./

# Install dependencies (no venv → installs into system site-packages)
RUN pdm config python.use_venv false && pdm install --prod --no-editable

# Copy your app
COPY . .

# Use the correct port your service listens on (8080 in your logs)
EXPOSE 8080

# Start the service (adjust path if your app lives in src/)
CMD ["pdm", "run", "python", "src/app.py"]