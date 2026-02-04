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


```bash


# Esegui le seguenti due istruzioni solo se vuoi configurare una nuova repo
dvc init 
dvc remote add origin https://dagshub.com/YOUR_USERNAME/YOUR_REPO.dvc 

# Configura remote DagshHub (sostituisci con le tue credenziali)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_USERNAME #username dagshub
dvc remote modify origin --local password YOUR_TOKEN #token personale dagshub

dvc pull #Scarica i dati dal cloud
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
pip install -r requirements.txt #Impiega ~10 minuti
```


## Addestramento
Per addestrare il modello tramite DVC, assicurati che i dati siano sincronizzati (`dvc pull`) e poi esegui:

```bash
dvc repro
```

Questo comando ricostruirà la pipeline di addestramento definita nei file `.dvc` e `dvc.yaml`, eseguendo solo gli step necessari in base alle modifiche rilevate. Gli output (modelli, metriche, artefatti) saranno tracciati e versionati automaticamente da DVC.

Al termine, puoi visualizzare lo stato della pipeline con:

```bash
dvc status
```

## Salvataggio dei progressi
Per salvare i progressi (modelli, dati, artefatti) su DagsHub e aggiornare la repository GitHub:

```bash
dvc push          # Carica i dati/artefatti su DagsHub
git add .         # Aggiungi i cambiamenti (inclusi i file .dvc aggiornati)
git commit -m "Salva progressi e aggiorna artefatti"
git push          # Aggiorna la repository su GitHub
```



## Esecuzione Locale (ANCORA IN FASE DI SVILUPPO)

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


## CI/CD

GitHub Actions esegue automaticamente:
- Linting e test
- Build Docker images
- Push artefatti su DVC (su push a main)
