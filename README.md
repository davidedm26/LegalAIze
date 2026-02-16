# LegalAIze: AI Audit Tool

LegalAIze is an auditing tool designed to assess technical documentation for compliance with the EU AI Act and ISO 42001 standards. It streamlines the evaluation process, helping organizations ensure their AI systems meet regulatory and quality requirements.

## Requirements

- Python 3.11 is required.
- Docker and Docker Compose must be installed.

## Technology Stack

- DVC: Artifacts versioning
- Qdrant: Vector database
- MLflow: experiments logging
- Sentence Transformers: Embedding model
- FastAPI: Backend API
- Streamlit: Frontend UI
- Docker: Containerization

## 1. Repository Initialization

Clone the repository:

```bash
git clone https://github.com/davidedm_26/LegalAIze.git
cd LegalAIze
```

Install DVC:

```bash
pip install dvc
```


## 2. MLflow Initialization


Create a `.env` file in the project root. By default, it is recommended to use a local MLflow instance for experiment tracking:

```
MLFLOW_TRACKING_URI=http://localhost:5000
```

If you use the local MLflow option, you must start the MLflow server before running experiments:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```
This will make the MLflow UI available at http://localhost:5000.

If you want to use a remote MLflow instance on DagsHub, you can create your own DagsHub repository and set the following in your `.env`:

```
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/YOUR_REPO.mlflow
DAGSHUB_USERNAME=YOUR_USERNAME
DAGSHUB_TOKEN=YOUR_TOKEN
```

If you want to collaborate with the LegalAIze team and log experiments to the main DagsHub repository, simply set your DagsHub username and token in the `.env`:

```
DAGSHUB_USERNAME=YOUR_USERNAME
DAGSHUB_TOKEN=YOUR_TOKEN
```

Note: 
- To log experiments to the main repository, you must request write access from the maintainers.
- You can refer to the .env.example file.

## 3. Artifact Initialization (Choose One Mode)

### A. Quick Demo Mode (uses precomputed artifacts)

Initialize DVC (required for new setups):

```bash
dvc init
dvc remote add origin https://dagshub.com/YOUR_USERNAME/YOUR_REPO.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_USERNAME
dvc remote modify origin --local password YOUR_TOKEN
```

Run DVC pull to download all required artifacts:
```bash
dvc pull
```

### B. Complete Demo Mode (recomputes all artifacts)

Run DVC repro to force full pipeline execution and artifact generation:
```bash
dvc repro --force
```

## 4. Container Build and Start

Build the containers:
```bash
docker compose build
```
Start the demo:
```bash
docker compose up
```

## 4. Running the Demo

- Backend: http://localhost:8000
- Frontend: http://localhost:8501

## Notes

- Python 3.11 is required for all scripts and containers.
- For local development, you can run backend and frontend separately as described in their respective folders.
- For CI/CD, GitHub Actions is configured for linting, testing, and Docker builds.
