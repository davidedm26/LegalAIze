FROM python:3.11-slim

WORKDIR /app

# Copia requirements e installa dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice frontend
COPY . .

# Esponi porta Streamlit
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
