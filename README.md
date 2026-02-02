# AISE Project - ML Pipeline

Progetto minimale che integra MLflow, DVC, FastAPI, Streamlit, Docker e GitHub Actions.

## Stack Tecnologico

- **MLflow**: Tracking esperimenti (via DagshHub)
- **DVC**: Gestione artefatti e dati (via DagshHub)
- **FastAPI**: Backend API
- **Streamlit**: Frontend UI
- **Docker**: Containerizzazione
- **GitHub Actions**: CI/CD

## Setup Iniziale

### 1. Configura DagshHub

```bash
# Inizializza DVC
dvc init

# Configura remote DagshHub (sostituisci con le tue credenziali)
dvc remote add origin https://dagshub.com/YOUR_USERNAME/YOUR_REPO.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_USERNAME
dvc remote modify origin --local password YOUR_TOKEN
```

### 2. Configura MLflow con DagshHub

Crea un file `.env`:
```
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow
MLFLOW_TRACKING_USERNAME=YOUR_USERNAME
MLFLOW_TRACKING_PASSWORD=YOUR_TOKEN
DAGSHUB_USER_TOKEN=YOUR_TOKEN
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

## Esecuzione Locale

### Backend (FastAPI)
```bash
cd backend
uvicorn app:app --reload --port 8000
```

### Frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```

## Esecuzione con Docker

```bash
docker-compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

## Train Model

```bash
python train_model.py
```

## CI/CD

GitHub Actions esegue automaticamente:
- Linting e test
- Build Docker images
- Push artefatti su DVC (su push a main)
