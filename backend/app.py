"""
FastAPI Backend - Minimal ML API
"""
import pickle
import os
from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

app = FastAPI(title="ML API", version="1.0.0")

# CORS per permettere richieste da Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica model da MLflow o da file locale
model = None

def load_model_from_mlflow():
    """Carica il modello più recente da MLflow"""
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")
        
        # Cerca l'ultimo run con successo
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Default")
        
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                run_id = runs[0].info.run_id
                model_uri = f"runs:/{run_id}/model"
                return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Errore caricamento da MLflow: {e}")
    return None

def load_model_from_file():
    """Fallback: carica da file locale"""
    MODEL_PATH = "../models/model.pkl"
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

# Prova a caricare il modello (prima da MLflow, poi da file locale)

# Funzione per caricare un modello specifico dal Model Registry MLflow
def load_model_from_registry(model_name: str, stage: str = "None"):
    """
    Carica un modello dal Model Registry MLflow/DagsHub
    model_name: nome del modello registrato
    stage: "None" per ultima versione, oppure "Staging", "Production", ecc.
    """
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

        client = mlflow.tracking.MlflowClient()
        # Recupera l'ultima versione del modello registrato
        if stage == "None":
            versions = client.get_latest_versions(model_name)
            if versions:
                model_uri = versions[0].source
                return mlflow.sklearn.load_model(model_uri)
        else:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if versions:
                model_uri = versions[0].source
                return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Errore caricamento dal Model Registry: {e}")
    return None

# Carica il modello come prima
model = load_model_from_mlflow()
if model is None:
    model = load_model_from_file()
    if model:
        print("✓ Modello caricato da file locale")
else:
    print("✓ Modello caricato da MLflow/DagShub")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: list[float]

@app.get("/")
def root():
    """Health check"""
    return {
        "status": "ok",
        "message": "ML API is running",
        "model_loaded": model is not None
    }

@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predizione ML
    
    Esempio request:
    {
        "features": [5.1, 3.5, 1.4, 0.2]
    }
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Train a model first."
        )
    
    try:
        # Prepara input
        X = np.array(request.features).reshape(1, -1)
        
        # Predizione
        prediction = int(model.predict(X)[0])
        probability = model.predict_proba(X)[0].tolist()
        
        return PredictionResponse(
            prediction=prediction,
            probability=probability
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
def model_info():
    """Informazioni sul modello"""
    if model is None:
        return {"error": "Model not loaded"}
    
    return {
        "type": type(model).__name__,
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
        "n_classes": model.n_classes_ if hasattr(model, 'n_classes_') else None
    }
