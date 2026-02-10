FROM python:3.11-slim
WORKDIR /app

# Install runtime deps
RUN pip install --no-cache-dir "qdrant-client>=1.7.0" requests

# Copy seeder script
COPY transfer_qdrant.py /app/transfer_qdrant.py
RUN chmod +x /app/transfer_qdrant.py

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "/app/transfer_qdrant.py"]
