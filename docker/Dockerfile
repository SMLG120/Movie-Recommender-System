FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8080
EXPOSE 8080

# Set Python path
ENV PYTHONPATH=/app:/app/src

# Start the application
CMD ["python", "src/app.py"]
