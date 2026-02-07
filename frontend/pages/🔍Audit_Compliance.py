import streamlit as st
import requests
import json
import time

# ==============================================================================
# CONFIGURAZIONE (HARDCODED - Il cliente non vede questo)
# ==============================================================================
BACKEND_URL = "http://localhost:8000"  # Indirizzo interno API
#BACKEND_URL = "http://backend:8000"  # Docker

st.set_page_config(
    page_title="Audit - LegalAIze",
    page_icon="🔍",
    layout="wide"
)

# Sidebar minimale (solo logo o info generiche, niente config tecnica)
with st.sidebar:
    #st.image("assets/logo.png", use_container_width=True) if st.sidebar.checkbox("Mostra Logo", True) else None
    st.markdown("### 📋 Stato Sistema")
    # Facciamo un check silenzioso al backend per mostrare un pallino verde/rosso
    try:
        if requests.get(f"{BACKEND_URL}/health", timeout=1).status_code == 200:
            st.success("Sistema Online")
        else:
            st.error("Manutenzione")
    except:
        st.error("Offline")
        
    st.info("ℹ️ Carica documenti PDF o TXT per ottenere un'analisi completa secondo gli standard ISO 42001 e EU AI Act.")

# ==============================================================================
# INTERFACCIA PRINCIPALE
# ==============================================================================
st.title("🔍 Compliance Audit")
st.markdown("Analisi automatica della documentazione tecnica.")

# 1. SEZIONE INPUT
with st.container(border=True):
    col_input, col_opt = st.columns([3, 1])
    
    with col_opt:
        input_mode = st.radio("Sorgente Dati:", ["📄 Carica File", "✍️ Incolla Testo"])
    
    with col_input:
        doc_text = ""
        files = None
        
        if input_mode == "✍️ Incolla Testo":
            doc_text = st.text_area(
                "Documentazione Tecnica", 
                height=150, 
                placeholder="Incolla qui la descrizione del sistema, l'architettura o la privacy policy..."
            )
        else:
            uploaded_file = st.file_uploader("Seleziona documento (PDF, TXT)", type=["pdf", "txt"])
            if uploaded_file:
                # Qui gestiresti la lettura del file. Per ora simuliamo che passi il testo o il file raw al backend
                # Se il tuo backend accetta file, useremmo 'files', se accetta testo, leggiamo il file qui.
                # Assumiamo per semplicità di leggere il testo qui per inviarlo al backend JSON:
                try:
                    if uploaded_file.type == "application/pdf":
                        st.info("⚠️ Lettura PDF simulata (implementare estrazione testo)")
                        doc_text = "Contenuto estratto dal PDF..." # Qui ci andrebbe pypdf
                    else:
                        doc_text = uploaded_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Errore lettura file: {e}")

# 2. AZIONE
analyze_btn = st.button("🚀 Avvia Analisi Compliance", type="primary", use_container_width=True)

# 3. OUTPUT REPORT
if analyze_btn and doc_text:
    
    with st.status("Analisi in corso...", expanded=True) as status:
        st.write("🧠 Generazione Report...")
        
        try:
            # CHIAMATA AL BACKEND
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                f"{BACKEND_URL}/audit", 
                json={"document_text": doc_text}, 
                headers=headers, 
                timeout=120
            )

            if response.status_code == 200:
                raw_data = response.json()
                req_list = raw_data.get("requirements", [])
                
                # --- CALCOLI SEMPLIFICATI (0-5) ---
                processed_reqs = []
                total_points = 0    # Somma dei voti (es. 3 + 4 + 5)
                max_points = 0      # Punteggio massimo possibile (es. 5 + 5 + 5)
                
                for item in req_list:
                    # 1. Recupero Dati (con i nomi corretti del backend)
                    r_name = item.get("Requirement_Name", "Requisito")
                    r_id = item.get("Mapped_ID", "N/A")
                    r_notes = item.get("Auditor_Notes", "Nessuna nota.")
                    
                    # 2. Gestione Punteggio (Semplice: è un numero 0-5)
                    score = item.get("Score", 0) 
                    
                    # Calcoliamo la percentuale per la barra (0-1.0)
                    progress_val = score / 5.0 
                    
                    # Aggiorniamo i totali per la media globale
                    total_points += score
                    max_points += 5
                    
                    processed_reqs.append({
                        "name": r_name,
                        "id": r_id,
                        "score_display": f"{score}/5", # Scritta "3/5"
                        "progress": progress_val,      # Valore barra (0.6)
                        "notes": r_notes
                    })

                # 3. CALCOLO MEDIA GLOBALE
                # Se ho fatto 80 punti su 100 disponibili -> 80%
                global_pct = (total_points / max_points) if max_points > 0 else 0.0

                status.update(label="Analisi completata!", state="complete", expanded=False)
                st.markdown("---")
                
                # --- VISUALIZZAZIONE ---
                
                # METRICHE IN ALTO
                c1, c2, c3 = st.columns(3)
                c1.metric("Punteggio Compliance", f"{global_pct*100:.0f}%", 
                          delta="Ottimo" if global_pct > 0.8 else "Migliorabile")
                c2.metric("Requisiti Analizzati", len(processed_reqs))
                c3.metric("Punti Totali", f"{total_points}/{max_points}")

                # LISTA REQUISITI
                st.subheader("📋 Dettaglio Requisiti")
                
                # Ordina: prima i voti bassi
                processed_reqs.sort(key=lambda x: x["progress"])

                for req in processed_reqs:
                    # Icona in base al voto (Maggiore di 3 su 5 è ok)
                    icon = "✅" if req["progress"] >= 0.6 else "⚠️" if req["progress"] >= 0.4 else "❌"
                    
                    with st.expander(f"{icon} {req['name']} ({req['score_display']})"):
                        col_A, col_B = st.columns([1, 2])
                        
                        with col_A:
                            st.caption(f"ID: {req['id']}")
                            st.progress(req["progress"]) # Barra verde
                        
                        with col_B:
                            st.caption("NOTE IA")
                            st.write(req["notes"])

            else:
                status.update(label="Errore Server", state="error")
                st.error(f"Errore HTTP: {response.status_code}")
                
        except Exception as e:
            status.update(label="Errore", state="error")
            st.error(f"Impossibile connettersi: {e}")

elif analyze_btn:
    st.warning("Carica un file o incolla del testo.")