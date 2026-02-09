FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice frontend
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit (0.0.0.0 to allow access from outside the container)
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"] 
