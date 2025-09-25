FROM python:3.9-slim-bullseye

WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port your service will run on
EXPOSE 8082

# Command to start your inference service
CMD ["python", "app.py"]