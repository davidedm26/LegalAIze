# AISE Project - ML Pipeline

Progetto minimale che integra MLflow, DVC, FastAPI, Streamlit, Docker e GitHub Actions.

## Stack Tecnologico

- **MLflow**: Tracking esperimenti (via DagsHub)
    <!--
    DagsHub permette di ospitare un istanza remota di mlflow. Questo consente a più collaboratori di lavorare nello stesso spazio degli esperimenti
    -->
- **DVC**: Gestione artefatti e dati (via DagsHub)
    <!--
    DVC (Data Version Control) è uno strumento open-source che consente di gestire versionamento, tracciamento e condivisione di dati e modelli nei progetti di machine learning. DVC sta a Dati come Git sta a Codice
    -->
- **FastAPI**: Backend API
- **Streamlit**: Frontend UI
- **Docker**: Containerizzazione
- **GitHub Actions**: CI/CD

## Setup Iniziale

### 1. Configura DVC (via DagsHub)

Se devi replicare il progetto 
```bash
# Inizializza DVC
dvc init 

# Configura remote DagshHub (sostituisci con le tue credenziali)
dvc remote add origin https://dagshub.com/YOUR_USERNAME/YOUR_REPO.dvc 
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_USERNAME
dvc remote modify origin --local password YOUR_TOKEN


```

Se devi collaborare al progetto
```bash
# Configura credenziali locali
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_USERNAME
dvc remote modify origin --local password YOUR_TOKEN

# Scaricare i dati (Dataset / VectorDB)
dvc pull
```

### 2. Configura MLflow con DagsHub

Crea un file `.env`:
```
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow
DAGSHUB_USERNAME=YOUR_USERNAME
DAGSHUB_TOKEN=YOUR_TOKEN
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
L'addestramento avviene automaticamente tramite Git Actions quando si pusha il nuovo codice, tuttavia se si vuole forzare il training locale si possono seguire questi passi:

- Effettua addestramento
```bash
dvc repro 
```
### Aggiorna Github/DagsHub
Salva il modello ottenuto su DagsHub 
```bash
dvc push 
```
Aggiorna file .dvc su github 
```bash
git add .
git commit -m "New trained model <add_tag>"
git push
```

## CI/CD

GitHub Actions esegue automaticamente:
- Linting e test
- Build Docker images
- Push artefatti su DVC (su push a main)
