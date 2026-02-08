FROM python:3.11-slim

WORKDIR /app

# Installa curl per l'healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice backend
COPY . .

# Crea directory per il modello
#RUN mkdir -p /app/models

# Esponi porta
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
