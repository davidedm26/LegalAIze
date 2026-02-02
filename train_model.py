"""
Minimal ML training pipeline con MLflow e DVC
"""
import os
import json
import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from dotenv import load_dotenv

# Carica variabili ambiente
load_dotenv()

# Configura MLflow per DagshHub
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

def load_params():
    """Carica parametri da params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def train():
    """Training pipeline minimale"""
    # Carica parametri
    params = load_params()
    
    # Carica dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, 
        data.target, 
        test_size=params['train']['test_size'],
        random_state=params['train']['random_state']
    )
    
    # Inizia MLflow run
    with mlflow.start_run():
        # Log parametri
        mlflow.log_params(params['model'])
        mlflow.log_params(params['train'])
        
        # Train model
        model = RandomForestClassifier(
            max_depth=params['model']['max_depth'],
            n_estimators=params['model']['n_estimators'],
            random_state=params['model']['random_state']
        )
        model.fit(X_train, y_train)
        
        # Predizioni e metriche
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metriche
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"✓ Model trained - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Salva model per DVC
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Salva metriche per DVC
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print("✓ Artifacts saved to models/ and metrics.json")

if __name__ == "__main__":
    train()
