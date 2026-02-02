"""
Streamlit Frontend - Minimal ML UI
"""
import streamlit as st
import requests
import json

# Configurazione
BACKEND_URL = "http://backend:8000"  # Docker
# BACKEND_URL = "http://localhost:8000"  # Locale

st.set_page_config(
    page_title="ML Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Prediction Demo")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurazione")
    backend_url = st.text_input("Backend URL", BACKEND_URL)
    st.markdown("---")
    st.markdown("### Info")
    st.markdown("Stack: MLflow, DVC, FastAPI, Streamlit, Docker")

# Check backend health
try:
    response = requests.get(f"{backend_url}/health", timeout=2)
    if response.status_code == 200:
        st.success("✅ Backend connesso")
        health_data = response.json()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", health_data.get("status", "N/A"))
        with col2:
            model_status = "✅ Loaded" if health_data.get("model_loaded") else "❌ Not Loaded"
            st.metric("Model", model_status)
    else:
        st.error("❌ Backend non disponibile")
except Exception as e:
    st.error(f"❌ Errore connessione backend: {str(e)}")
    st.info("💡 Assicurati che il backend sia in esecuzione")

st.markdown("---")

# Prediction Section
st.header("📊 Fai una Predizione")

st.markdown("**Inserisci le features (Iris Dataset):**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
with col3:
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
with col4:
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("🔮 Predici", type="primary"):
    try:
        # Prepara request
        payload = {
            "features": [sepal_length, sepal_width, petal_length, petal_width]
        }
        
        # Chiamata API
        with st.spinner("Predizione in corso..."):
            response = requests.post(
                f"{backend_url}/predict",
                json=payload,
                timeout=5
            )
        
        if response.status_code == 200:
            result = response.json()
            
            st.success("✅ Predizione completata!")
            
            # Risultati
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("🎯 Classe Predetta", result["prediction"])
            
            with col2:
                st.markdown("**📈 Probabilità:**")
                probs = result["probability"]
                for i, prob in enumerate(probs):
                    st.write(f"Classe {i}: {prob:.4f}")
            
            # Visualizza JSON
            with st.expander("📄 Risposta JSON"):
                st.json(result)
        else:
            st.error(f"❌ Errore: {response.status_code}")
            st.write(response.text)
    
    except Exception as e:
        st.error(f"❌ Errore durante la predizione: {str(e)}")

# Model Info
st.markdown("---")
if st.button("ℹ️ Info Modello"):
    try:
        response = requests.get(f"{backend_url}/model/info", timeout=2)
        if response.status_code == 200:
            st.json(response.json())
    except Exception as e:
        st.error(f"Errore: {str(e)}")
