"""
Streamlit Frontend - LegalAIze Audit UI
"""
import streamlit as st
import requests
import json

# Configurazione
BACKEND_URL = "http://backend:8000"  # Docker

st.set_page_config(
    page_title="LegalAIze Audit Tool",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LegalAIze: Strumento di Audit Normativo")
st.markdown("Verifica la compliance della tua documentazione rispetto a **AI Act**, **GDPR** e standard **ISO**.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurazione")
    backend_url = st.text_input("Backend URL", BACKEND_URL)
    st.markdown("---")
    st.markdown("### Info Progetto")
    st.markdown("- **Pipeline**: DVC")
    st.markdown("- **Database**: Qdrant")
    st.markdown("- **Modello**: all-MiniLM-L6-v2")

# Check backend health
try:
    response = requests.get(f"{backend_url}/health", timeout=2)
    if response.status_code == 200:
        st.success("✅ Backend connesso")
        health_data = response.json()
        rag_status = "✅ Pronta" if health_data.get("rag_ready") else "❌ Non inizializzata"
        st.metric("Ricerca Normativa", rag_status)
    else:
        st.error("❌ Backend non disponibile")
except Exception as e:
    st.error(f"❌ Errore connessione backend: {str(e)}")

st.markdown("---")

# Audit Section
st.header("🔍 Avvia Audit Compliance")

# Scelta input
input_mode = st.radio("Metodo di input:", ["Inserimento Testo", "Caricamento Documento (Simulato)"])

if input_mode == "Inserimento Testo":
    doc_text = st.text_area("Incolla qui la descrizione del sistema o la documentazione tecnica:", height=200, placeholder="Esempio: Il sistema utilizza algoritmi di riconoscimento facciale per l'identificazione in tempo reale in spazi pubblici...")
else:
    uploaded_file = st.file_uploader("Carica specifica tecnica (PDF, TXT)", type=["pdf", "txt"])
    doc_text = "Contenuto del file caricato..." if uploaded_file else ""

if st.button("🚀 Esegui Audit", type="primary"):
    if not doc_text or doc_text == "Contenuto del file caricato...":
        st.warning("Ehi, inserisci del testo per l'analisi!")
    else:
        try:
            with st.spinner("Analisi della compliance in corso..."):
                # Chiamata all'endpoint audit
                payload = {"document_text": doc_text}
                response = requests.post(
                    f"{backend_url}/audit",
                    json=payload,
                    timeout=20
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Visualizzazione Risultati
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    score = result["compliance_score"]
                    st.metric("Compliance Score", f"{score*100:.1f}%")
                    if score > 0.7:
                        st.success("Altà conformità rilevata")
                    elif score > 0.5:
                        st.warning("Conformità parziale")
                    else:
                        st.error("Possibili criticità rilevate")
                
                with col2:
                    st.subheader("💡 Raccomandazioni")
                    st.info(result["recommendations"])
                
                st.markdown("---")
                st.subheader("📄 Riferimenti Normativi Trovati")
                
                for i, finding in enumerate(result["findings"]):
                    with st.expander(f"Rif {i+1}: {finding['source']} (Score: {finding['score']:.4f})"):
                        st.write(finding['content'])
            else:
                st.error(f"Errore API: {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error(f"Errore durante l'audit: {str(e)}")

st.markdown("---")
st.caption("LegalAIze Audit Tool - AISE MSc Unina")
