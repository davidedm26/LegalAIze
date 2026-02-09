import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="LegalAIze - AI Act & ISO Compliance",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    /* HIDE DEFAULT STREAMLIT NAV */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* TOP SPACING */
    .block-container { 
        padding-top: 3rem; 
        padding-bottom: 2rem; 
    }
    
    /* Vertical alignment */
    div[data-testid="stColumn"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* HERO TITLE */
    .hero-title-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        color: #1e3a8a; /* Deep Navy Blue */
        margin: 0;
        padding: 0;
        line-height: 1.2;
        white-space: nowrap; 
    }

    /* HERO SUBTITLE */
    .hero-subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        font-weight: 400;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 40px;
    }

    /* --- THE GIANT LENS FIX --- */
    
    /* 1. THE CONTAINER (The clickable box) */
    a[href*="Audit_Compliance"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        text-decoration: none !important;
        
        /* ALTEZZA FISSA: Diamo tanto spazio verticale per evitare tagli */
        min-height: 260px !important; 
        width: 100% !important;
        
        /* Flexbox per centrare */
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        
        overflow: visible !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* 2. THE EMOJI ITSELF */
    a[href*="Audit_Compliance"] p {
        font-size: 8rem !important; 
        line-height: 1.5 !important; 
        margin: 0 !important;
        padding: 0 !important;
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); 
        cursor: pointer;
        display: inline-block;
        vertical-align: middle;
        position: relative;
        z-index: 10;
    }

    /* Hover Effect */
    a[href*="Audit_Compliance"]:hover p {
        transform: scale(1.15) translateY(-10px); 
        filter: drop-shadow(0 20px 20px rgba(30, 58, 138, 0.2));
    }
    
    /* DESCRIPTION TEXT BELOW LENS */
    .lens-description {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        /* Margine negativo per "tirare su" il testo vicino alla lente */
        margin-top: -20px; 
        position: relative;
        z-index: 5;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .lens-cta {
        font-weight: 800;
        color: #1e3a8a; /* Blu scuro come il titolo */
        font-size: 1.8rem; /* Bello grande */
        margin-bottom: 10px;
        letter-spacing: -0.5px;
    }
    
    /* Sidebar Styling text */
    .sidebar-text {
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 15px;
        line-height: 1.6;
    }

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
# SIDEBAR CONTENT (ENGLISH)
# ==============================================================================
with st.sidebar:
    st.markdown("### 🤖 How does it work?")
    st.markdown("""
    <div class="sidebar-text">
    Welcome to <b>LegalAIze</b>. This tool assists you in verifying the compliance of your AI systems.
    <br><br>
    <b>1. Upload Documents</b><br>
    Provide technical documentation or policy files (PDF/TXT).
    <br><br>
    <b>2. AI Analysis</b><br>
    The system cross-references data with the <b>EU AI Act</b> and <b>ISO 42001</b> standards.
    <br><br>
    <b>3. Get Report</b><br>
    Receive a detailed analysis of risks and regulatory discrepancies.
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Note:** This is an AI-based decision support tool and does not replace professional legal advice.")

# ==============================================================================
# HEADER (LOGO + TITLE)
# ==============================================================================
col_spacer_L, col_center, col_spacer_R = st.columns([1, 3, 1]) 

with col_center:
    c_logo, c_text = st.columns([1, 3]) 
    
    with c_logo:
        # SCALE ICON
        st.markdown(
            '<div style="font-size: 5rem; text-align: right; line-height: 1;">⚖️</div>', 
            unsafe_allow_html=True
        )
        
    with c_text:
        st.markdown('<div class="hero-title-text">LegalAIze</div>', unsafe_allow_html=True)

# Subtitle
st.markdown("""
<div class="hero-subtitle">
    Your intelligent consultant for <b>AI Act</b> and <b>ISO</b> standard compliance.<br>
    Navigate regulatory complexities with confidence and precision.
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# MAIN SECTION (THE CLICKABLE LENS)
# ==============================================================================
st.write("") 

col_left, col_mid, col_right = st.columns([1, 2, 1])

with col_mid:
    # 1. THE CLICKABLE LENS
    # Il link contiene solo l'emoji. Tutto il resto è gestito dal CSS.
    st.page_link("pages/Audit_Compliance.py", label="🔍", use_container_width=True)

    # 2. DESCRIPTION TEXT (Subito sotto)
    st.markdown("""
    <div class="lens-description">
        <div class="lens-cta">Click to Start Audit</div>
        Our AI engine is ready to analyze your technical documentation<br>
        against <b>EU AI Act</b> & <b>ISO 42001</b> standards.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="footer">
    LegalAIze Project • AISE MSc Unina • Powered by RAG & LLMs
</div>
""", unsafe_allow_html=True)