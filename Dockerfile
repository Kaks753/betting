FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ensure data directory exists (will be overridden by Fly volume mount)
RUN mkdir -p /data

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "kbet.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
