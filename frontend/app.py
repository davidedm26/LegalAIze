import streamlit as st

# ==============================================================================
# CONFIGURAZIONE PAGINA
# ==============================================================================
st.set_page_config(
    page_title="LegalAIze - AI Act & ISO Compliance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CSS CUSTOM
# ==============================================================================
st.markdown("""
<style>
    /* SPAZIO SUPERIORE */
    .block-container { 
        padding-top: 5rem; 
        padding-bottom: 2rem; 
    }
    
    /* Allineamento verticale elementi nelle colonne */
    div[data-testid="stColumn"] {
        display: flex;
        align-items: center;
    }

    /* Stile del Titolo */
    .hero-title-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        color: #1e3a8a;
        margin: 0;
        padding: 0;
        line-height: 1.2;
        white-space: nowrap; 
    }

    /* Sottotitolo */
    .hero-subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        font-weight: 400;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    
    /* AUDIT CARD */
    .audit-card {
        background-color: white;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        text-align: center;
        transition: transform 0.2s;
        margin-bottom: 20px;
    }
    .audit-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .card-icon { font-size: 3rem; margin-bottom: 15px; }
    .card-title { font-size: 1.5rem; font-weight: 700; color: #111827; margin-bottom: 10px; }
    .card-text { font-size: 1rem; color: #6b7280; margin-bottom: 25px; }

    /* FOOTER */
    .footer {
        text-align: center;
        margin-top: 80px;
        color: #9ca3af;
        font-size: 0.8rem;
        border-top: 1px solid #e5e7eb;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HEADER (LOGO + TITOLO)
# ==============================================================================

# 1. GABBIA ESTERNA (Posizione generale)
# Ho messo [1, 3, 1]. Il "3" centrale dà molto più spazio.
# Se vuoi spostare tutto a DESTRA, aumenta il primo numero (es. [1.5, 3, 0.5])
# Se vuoi spostare tutto a SINISTRA, diminuisci il primo numero (es. [0.5, 3, 1.5])
col_spacer_L, col_center, col_spacer_R = st.columns([1, 3, 1]) # <--- MODIFICA QUI PER SPOSTARE TUTTO IL BLOCCO

with col_center:
    # 2. GABBIA INTERNA (Spazio tra Logo e Testo)
    # [1.2, 3] dà un po' più di spazio alla colonna del logo così non lo taglia
    c_logo, c_text = st.columns([1.2, 3]) # <--- MODIFICA QUI SE IL TESTO E' TROPPO VICINO O LONTANO
    
    with c_logo:
        # 3. DIMENSIONE LOGO
        # Qui decidi quanto è grande l'immagine.
        # Se aumenti questo numero, devi assicurarti che la colonna "c_logo" (punto 2) sia abbastanza larga.
        st.image("assets/logo.png", width=120) # <--- MODIFICA QUI PER INGRANDIRE/RIMPICCIOLIRE
        
    with c_text:
        st.markdown('<div class="hero-title-text">LegalAIze</div>', unsafe_allow_html=True)

# Sottotitolo
st.markdown("""
<div class="hero-subtitle">
    Il tuo consulente intelligente per la conformità <b>AI Act</b> e standard <b>ISO</b>.<br>
    Naviga le complessità normative con sicurezza e precisione.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# SEZIONE PRINCIPALE (Audit Compliance)
# ==============================================================================
st.write("") 

col_left, col_mid, col_right = st.columns([1, 1.5, 1])

with col_mid:
    st.markdown("""
    <div class="audit-card">
        <div class="card-icon">🔍</div>
        <div class="card-title">Compliance Audit</div>
        <div class="card-text">
            Carica la documentazione tecnica o la Privacy Policy.
            L'IA analizzerà i rischi e genererà un report dettagliato.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.page_link("pages/🔍Audit_Compliance.py", label="Avvia Audit", icon="🚀", use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer">
    LegalAIze Project • AISE MSc Unina • Powered by RAG & LLMs
</div>
""", unsafe_allow_html=True)