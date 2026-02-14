FROM python:3.11-slim  

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code (FastAPI app + rag engine)
COPY app.py rag_engine.py ./

# Create directory for the model
RUN mkdir -p /app/models

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
