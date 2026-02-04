# LegalAIze - AI Audit Tool

Strumento di **Audit Normativo** per la verifica della conformità di documentazione tecnica rispetto a **AI Act**, **GDPR** e standard **ISO**.

## Stack Tecnologico

- **DVC**: Gestione artefatti e versionamento dell'indice vettoriale.
- **Qdrant**: Vector Database (Local Mode) per la ricerca semantica.
- **Sentence Transformers**: Modello `all-MiniLM-L6-v2` per gli embedding.
- **FastAPI**: Backend per l'esecuzione dell'audit.
- **Streamlit**: Frontend UI per il caricamento dei documenti e reportistica.
- **Docker**: Containerizzazione dell'intero stack.

## Setup Iniziale

### 1. Configura DVC (via DagsHub)


```bash
# Installa dvc
pip install dvc

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

## Esecuzione Locale [IN FASE DI SVILUPPO]

#### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

#### Frontend (Streamlit)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Esecuzione con Docker [IN FASE DI SVILUPPO]

```bash
docker-compose up --build
```

- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:8501



## CI/CD

GitHub Actions esegue automatically:
- Linting e Testing
- Docker Build
