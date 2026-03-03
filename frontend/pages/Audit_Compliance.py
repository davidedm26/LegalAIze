import streamlit as st
import requests
import json
import os
import time
from fpdf import FPDF
from pypdf import PdfReader
import plotly.graph_objects as go
import streamlit as st
import requests
import json
import os
import time
from fpdf import FPDF
from pypdf import PdfReader
import plotly.graph_objects as go

# ==============================================================================
# CONFIGURATION & DYNAMIC PATHS
# ==============================================================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- DYNAMIC PATH CALCULATION ---
# This ensures the code works on any computer (yours or your colleague's).
# It calculates the path relative to where this script file is located.
try:
    # 1. Get the directory of this script (e.g., .../LegalAIze/frontend/pages)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up two levels to reach the project root 'LegalAIze'
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    
    # 3. Build paths to data folders
    MAPPING_FILE = os.path.join(project_root, "data", "mapping.json")
    REQUIREMENTS_FILE = os.path.join(project_root, "data", "processed", "requirement_chunks.json")

    print("Loaded MAPPING_FILE from:", MAPPING_FILE)
    print("Loaded REQUIREMENTS_FILE from:", REQUIREMENTS_FILE)

except Exception as e:
    # Fallback just in case
    MAPPING_FILE = "mapping.json"
    REQUIREMENTS_FILE = "requirement_chunks.json"
    print(f"Path Error: {e}")

# ==============================================================================
# PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Audit - LegalAIze",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"  # <--- SIDEBAR NOW OPEN BY DEFAULT
)

# --- SESSION STATE & RESET LOGIC ---
if 'audit_results' not in st.session_state:
    st.session_state.audit_results = None
if 'global_score' not in st.session_state:
    st.session_state.global_score = 0
if 'total_points' not in st.session_state:
    st.session_state.total_points = 0
if 'max_points' not in st.session_state:
    st.session_state.max_points = 0

# Function to clear results automatically when a new file is uploaded
def reset_session():
    st.session_state.audit_results = None
    st.session_state.global_score = 0
    st.session_state.total_points = 0
    st.session_state.max_points = 0

# Generic function to load any JSON file
@st.cache_data
def load_json_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON Error loading {filepath}: {e}")
        return {}

# Load both files into memory
MAPPING_DATA = load_json_data(MAPPING_FILE)
REQUIREMENTS_DATA = load_json_data(REQUIREMENTS_FILE)



# ==============================================================================
# CUSTOM CSS
# ==============================================================================
st.markdown("""
<style>
    /* HIDE SIDEBAR NAV */
    [data-testid="stSidebarNav"] { display: none; }
    
    /* FOOTER STYLE */
    .legal-disclaimer {
        font-size: 0.75rem;
        color: var(--text-color-secondary, #6b7280);
        text-align: center;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid var(--secondary-background-color, #e5e7eb);
        font-style: italic;
    }
    
    /* TOP SPACING */
    .block-container { padding-top: 2rem; }

    /* --- BUTTON STYLING (Matching Home Page Blue) --- */
    div.stButton > button:first-child, 
    div.stDownloadButton > button:first-child {
        background-color: var(--primary-color, #1e3a8a) !important;
        color: var(--button-text-color, white) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }

    div.stButton > button:first-child:hover,
    div.stDownloadButton > button:first-child:hover {
        background-color: var(--primary-color-dark, #172554) !important;
        color: var(--button-text-color-hover, #e2e8f0) !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* TEXT HIGHLIGHTS */
    h1, h2, h3 {
        color: var(--primary-color, #1e3a8a) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HELPER: PDF GENERATION
# ==============================================================================
def create_pdf_report(requirements, global_score, total_points, max_points):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.set_text_color(30, 58, 138) 
            self.cell(0, 10, 'LegalAIze - Compliance Audit Report', 0, 1, 'C')
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, 25, 200, 25)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            self.cell(0, 5, "Not official legal advice.", 0, 0, 'R')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # --- SUMMARY ---
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.write(10, "Global Compliance Score: ")
    if global_score >= 0.8: pdf.set_text_color(0, 150, 0)
    elif global_score >= 0.5: pdf.set_text_color(255, 140, 0)
    else: pdf.set_text_color(200, 0, 0)
    pdf.write(10, f"{global_score*100:.0f}%\n")
    
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Total Points: {total_points}/{max_points}", ln=True)
    pdf.cell(0, 8, f"Framework: AI Act + ISO 42001", ln=True)
    pdf.ln(10)

    # --- REQUIREMENTS GROUPED BY PRINCIPLE ---
    pdf.set_font("Arial", 'B', 14) 
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Detailed Requirements Analysis", ln=True)
    pdf.ln(5)

    # Group requirements by principle (reuse group_requirements_by_principle)
    grouped = group_requirements_by_principle(requirements, MAPPING_DATA)
    for principle, reqs in grouped.items():
        # Calculate overall score for this principle
        if reqs:
            total_score = sum(r.get('progress', 0) * 5 for r in reqs)
            max_score = len(reqs) * 5
            avg_score = total_score / max_score if max_score > 0 else 0
            avg_score_5 = (total_score / len(reqs)) if len(reqs) > 0 else 0
        else:
            avg_score = 0
            avg_score_5 = 0
        pdf.set_font("Arial", 'B', 13)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 9, f"Ethical Principle: {principle}", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(30, 58, 138)
        pdf.cell(0, 7, f"Overall Score: ({avg_score_5:.2f}/5)", ln=True)
        pdf.ln(2)
        for req in reqs:
            def clean_text(text):
                if not text: return ""
                txt_fixed = text.replace("—", "-").replace("–", "-").replace("’", "'").replace('“', '"').replace('”', '"')
                return str(txt_fixed).encode('latin-1', 'replace').decode('latin-1')

            name = clean_text(req['name'])
            notes = clean_text(req['notes'])
            rationale = clean_text(req.get('rationale', ''))

            mapping_info = get_mapping_info(MAPPING_DATA, req.get('id'))
            iso_ref = clean_text(", ".join(mapping_info.get("iso_42001_sections", [])) if mapping_info.get("iso_42001_sections") else "N/A")
            ai_articles = mapping_info.get("eu_ai_act_articles", [])
            ai_refs_str = ", ".join(ai_articles) if ai_articles else "N/A"
            ai_refs_clean = clean_text(ai_refs_str)

            pdf.set_font("Arial", 'B', 11)
            pdf.set_text_color(30, 58, 138)
            pdf.cell(0, 8, f"[{req['score_display']}] {name}", ln=True)

            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(80, 80, 80)
            pdf.cell(0, 5, f"ID: {req['id']}", ln=True)
            pdf.cell(0, 5, f"ISO Ref: {iso_ref}", ln=True)
            pdf.cell(0, 5, f"AI Act: {ai_refs_clean}", ln=True)

            pdf.ln(2)
            pdf.set_font("Arial", size=10)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 6, f"Findings: {notes}")
            pdf.ln(5)
            pdf.set_draw_color(240, 240, 240)
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)

            pdf.set_font("Arial", 'I', 9)
            pdf.set_text_color(100, 100, 100)
            pdf.multi_cell(0, 5, f"Rationale: {rationale}")
            pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1', 'replace')

def get_mapping_info(mapping_data, requirement_id):
    principles = mapping_data.get("eu_ai_act_ethical_principle", [])
    for principle in principles:
        for req in principle.get("technical_requirements", []):
            if req.get("id") == requirement_id:
                return req
    return {}

def group_requirements_by_principle(requirements, mapping_data):
    grouped = {}
    for req in requirements:
        principle = None
        for p in mapping_data.get("eu_ai_act_ethical_principle", []):
            for r in p.get("technical_requirements", []):
                if r.get("id") == req.get("id"):
                    principle = p.get("ethical_principle")
                    break
            if principle:
                break
        if principle:
            grouped.setdefault(principle, []).append(req)
    return grouped

def get_reference_details(req_id, requirements_data):
    """
    Estrae i contenuti di dettaglio per le reference (AI Act e ISO) 
    dal file requirement_chunks.json basandosi sull'ID.
    """
    # Cerca il chunk corrispondente all'ID
    req_chunk = next((r for r in requirements_data if r.get("id") == req_id), None)
    
    ai_act_dict = {}
    iso_dict = {}
    
    if req_chunk:
        # Estrai articoli EU AI Act
        for ai_ref in req_chunk.get("euAiActArticles", []):
            ai_act_dict[ai_ref.get("reference")] = ai_ref.get("content", "Nessun contenuto disponibile.")
            
        # Estrai controlli ISO 42001
        for iso_ref in req_chunk.get("iso42001Reference", []):
            content = iso_ref.get("content", "")
            # Gestisci il caso in cui la ISO usi "control" e "implementation_guidance" invece di "content"
            if not content:
                control = iso_ref.get("control", "")
                guidance = iso_ref.get("implementation_guidance", "")
                content = f"**Control:**\n{control}\n\n**Guidance:**\n{guidance}".strip()
            iso_dict[iso_ref.get("reference")] = content
            
    return ai_act_dict, iso_dict

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### 📋 System Status")
    try:
        if requests.get(f"{BACKEND_URL}/health", timeout=1).status_code == 200:
            st.success("Online")
        else:
            st.error("Maintenance")
    except:
        st.error("Offline")
    
    st.markdown("---")
    
    # Legend Section
    st.markdown("### ℹ️ Legend")
    st.caption("Compliance levels:")
    st.markdown("✅ **High Compliance** (>80%)")
    st.markdown("⚠️ **Partial Compliance** (40-80%)")
    st.markdown("❌ **Non-Compliant** (<40%)")
    
    st.markdown("---")
    st.info("Upload PDF or TXT documents for **AI Act** & **ISO 42001** analysis.")
    
    # Manual Reset Button
    if st.button("🔄 Clear Analysis"):
        reset_session()
        st.rerun()

# ==============================================================================
# MAIN UI
# ==============================================================================


# --- BACK TO HOME BUTTON IN ALTO A DESTRA ---

# --- Migliore allineamento Home in alto a destra ---
col_spacer, col_home = st.columns([12, 1])
with col_home:
    st.markdown("<div style='padding-top:0.5rem; padding-right:0.5rem;'>", unsafe_allow_html=True)
    st.page_link("app.py", label="Home", icon="🏠", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.title("🔍 Audit Compliance")
st.markdown("Automated analysis of technical documentation powered by AI.")

if not MAPPING_DATA:
    st.warning(f"⚠️ Warning: 'mapping.json' not found at expected path: {MAPPING_FILE}")

# 1. Grounded SECTION
with st.container(border=True):
    col_input, col_opt = st.columns([3, 1])
    with col_opt:
        input_mode = st.radio("Data Source:", ["📄 Upload File", "✍️ Paste Text"], on_change=reset_session)
    with col_input:
        doc_text = ""
        if input_mode == "✍️ Paste Text":
            # on_change=reset_session -> Clears analysis when text changes
            doc_text = st.text_area("Technical Documentation", height=150, placeholder="Paste text here...", on_change=reset_session)
        else:
            # on_change==session -> Clears analysis when file changes
            uploaded_file = st.file_uploader("Select document (PDF, TXT)", type=["pdf", "txt"], on_change=reset_session)
            if uploaded_file:
                try:
                    if uploaded_file.type == "application/pdf":
                        reader = PdfReader(uploaded_file)
                        text_content = []
                        for page in reader.pages:
                            text_content.append(page.extract_text())
                        doc_text = "\n".join(text_content)
                    else:
                        doc_text = uploaded_file.read().decode("utf-8")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

# 2. ACTION BUTTON
analyze_btn = st.button("🚀 Start Compliance Analysis", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------
# ANALYSIS LOGIC
# ------------------------------------------------------------------------------
if analyze_btn and doc_text:
    with st.status("🧠 Analyzing document with AI...", expanded=True) as status:
        try:
            st.write("📤 Generating Report...")
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                f"{BACKEND_URL}/audit", 
                json={"document_text": doc_text}, 
                headers=headers, timeout=180
            )

            if response.status_code == 200:
                st.write("📥 Receiving analysis...")
                raw_data = response.json()
                req_list = raw_data.get("requirements", [])
                
                processed_reqs = []
                total = 0    
                maxim = 0      
                
                st.write("📊 Calculating compliance scores...")
                for item in req_list:
                    r_name = item.get("Requirement_Name", "Unnamed Requirement")
                    r_id = item.get("Requirement_ID", "N/A")
                    r_notes = item.get("Auditor_Notes", "No notes available.")
                    r_rationale = item.get("Rationale", "No rationale provided.")
                    r_sub_reqs = item.get("SubRequirements", [])
                    score = item.get("Score", 0) 
                    
                    processed_reqs.append({
                        "name": r_name, "id": r_id, "score_display": f"{score}/5", 
                        "progress": score / 5.0, "notes": r_notes, "rationale": r_rationale,
                        "sub_requirements": r_sub_reqs
                    })
                    total += score
                    maxim += 5

                # Save to Session State
                st.session_state.audit_results = processed_reqs
                st.session_state.total_points = total
                st.session_state.max_points = maxim
                st.session_state.global_score = (total / maxim) if maxim > 0 else 0.0
                
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                time.sleep(1) 
                st.rerun() 
                
            else:
                status.update(label="❌ Error", state="error")
                st.error(f"Server Error: HTTP {response.status_code}")
        except Exception as e:
            status.update(label="❌ Connection Failed", state="error")
            st.error(f"Connection failed: {e}")

# ------------------------------------------------------------------------------
# OUTPUT VISUALIZATION
# ------------------------------------------------------------------------------
if st.session_state.audit_results is not None:
    
    results = st.session_state.audit_results
    glob_score = st.session_state.global_score
    
    with st.expander("✅ Analysis Results (Report Complete)", expanded=True):
        
        # --- METRICS ---
        c1, c2, c3 = st.columns(3)
        c1.metric("Compliance Score", f"{glob_score*100:.0f}%", 
                  delta="Excellent" if glob_score > 0.8 else "Needs Work")
        c2.metric("Requirements Analyzed", len(results))
        c3.metric("Total Points", f"{st.session_state.total_points}/{st.session_state.max_points}")

        st.markdown("---")

        # --- HEXAGONAL RADAR CHART PRINCIPLES ---
        # Emoji mapping for principles
        principle_emojis = {
            "Human Agency & Oversight": "👤",
            "Technical Robustness and Safety": "🛡️",
            "Privacy & Data Governance": "🔐",
            "Transparency": "🔎",
            "Diversity, Non-discrimination & Fairness": "🌈",
            "Social & Environmental Well-being": "🌍",
        }
        grouped = group_requirements_by_principle(results, MAPPING_DATA)
        principles = list(grouped.keys())
        scores = []
        labels = []
        for principle in principles:
            reqs = grouped[principle]
            avg = sum(r['progress'] for r in reqs) / len(reqs) if reqs else 0
            scores.append(avg)
            emoji = principle_emojis.get(principle, "🟦")
            labels.append(f"{emoji} {principle}")
        #  (radar chart)
        if principles:
            labels.append(labels[0])
            # Scale scores to 0-5 for the chart
            scores_scaled = [s * 5 for s in scores]
            scores_scaled.append(scores_scaled[0])
            fig = go.Figure(data=go.Scatterpolar(
                r=scores_scaled,
                theta=labels,
                fill='toself',
                line=dict(color='#1e3a8a'),
                marker=dict(color='#1e3a8a')
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 5],
                        tickvals=[0, 1, 2, 3, 4, 5],
                        tickfont=dict(size=14, color='#1e3a8a'),
                        gridcolor='#b6c6e3',
                        gridwidth=1
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=16, color='#1e3a8a')
                    )
                ),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            

        st.markdown("---")
        
        # --- PDF DOWNLOAD ---
        c_pdf_left, c_pdf_mid, c_pdf_right = st.columns([1, 2, 1])
        with c_pdf_mid:
            # Generate PDF only once per session to avoid rerun on download
            if 'pdf_bytes' not in st.session_state:
                st.session_state.pdf_bytes = create_pdf_report(
                    results, glob_score, 
                    st.session_state.total_points, st.session_state.max_points
                )
            st.download_button(
                label="📄 DOWNLOAD FULL PDF REPORT", 
                data=st.session_state.pdf_bytes,
                file_name="compliance_report.pdf",
                mime="application/pdf",
                type="primary", 
                use_container_width=True,
                key="download_pdf_button",
                icon="📥"
            )

        st.markdown("---")
        
        # --- REQUIREMENTS DETAIL GROUPED BY PRINCIPLE ---
        st.subheader("📋 Requirements Detail (by Ethical Principle)")
        grouped_detail = group_requirements_by_principle(results, MAPPING_DATA)
        for principle, reqs in grouped_detail.items():
            # Calculate overall score for this principle
            if reqs:
                total_score = sum(r.get('progress', 0) * 5 for r in reqs)
                max_score = len(reqs) * 5
                avg_score = total_score / max_score if max_score > 0 else 0
                avg_score_5 = (total_score / len(reqs)) if len(reqs) > 0 else 0
            else:
                avg_score = 0
                avg_score_5 = 0
            emoji = principle_emojis.get(principle, "🟦")
            st.markdown(f"### {emoji} {principle}")
            st.markdown(f"**Overall Score:** ({avg_score_5:.2f}/5)")
            reqs_sorted = sorted(reqs, key=lambda x: x["progress"])
            for req in reqs_sorted:
                if req["progress"] >= 0.8:
                    icon = "✅"
                elif req["progress"] >= 0.4:
                    icon = "⚠️"
                else:
                    icon = "❌"

                map_info = get_mapping_info(MAPPING_DATA, req.get('id'))
                iso_ui_list = map_info.get("iso_42001_sections", [])
                ai_act_list = map_info.get("eu_ai_act_articles", [])

                iso_ui = ", ".join(iso_ui_list) if iso_ui_list else "N/A"
                ai_act_ui = ", ".join(ai_act_list) if ai_act_list else "N/A"
                if not ai_act_ui:
                    ai_act_ui = "N/A"
                if not iso_ui:
                    iso_ui = "N/A"

                with st.expander(f"{icon} {req['name']} ({req['score_display']})"):
                    col_A, col_B = st.columns([1, 2])
                    with col_A:
                        st.caption("DETAILS")
                        st.markdown(f"**ID:** `{req['id']}`")
                        st.markdown(f"**Score:** {req['score_display']}")

                        st.caption("REFERENCES")
                        
                        # --- LOGICA TOOLTIP ---
                        # Recupera i dizionari con i testi lunghi per l'ID corrente
                        ai_act_dict, iso_dict = get_reference_details(req['id'], REQUIREMENTS_DATA)

                        # Mostra ISO 42001 in verticale con tooltip
                        if iso_ui_list:
                            st.markdown("**ISO 42001:**")
                            for ref in iso_ui_list:
                                help_text = iso_dict.get(ref, "Dettagli non trovati nel JSON.")
                                st.markdown(f"- {ref}", help=help_text)
                        else:
                            st.markdown("**ISO 42001:** N/A")
                        
                        st.write("") # extra spacing
                        
                        # Mostra EU AI Act in verticale con tooltip
                        if ai_act_list:
                            st.markdown("**EU AI Act:**")
                            for ref in ai_act_list:
                                help_text = ai_act_dict.get(ref, "Dettagli non trovati nel JSON.")
                                st.markdown(f"- {ref}", help=help_text)
                        else:
                            st.markdown("**EU AI Act:** N/A")
                        # ----------------------------

                        st.progress(req["progress"])
                        
                    with col_B:
                        st.caption("AI FINDINGS")
                        st.write(req["notes"])

                        st.caption("Rationale")
                        st.text(req.get("rationale", "No rationale provided."))

                        sub_reqs = req.get("sub_requirements", [])
                        if sub_reqs:
                            with st.expander("Show Sub-Requirements Details"):
                                for sub in sub_reqs:
                                    st.markdown(f"#### {sub.get('Source', 'N/A')} - {sub.get('Reference', 'N/A')}")
                                    st.markdown(f"**Score:** {sub.get('Score', 'N/A')}")
                                    st.markdown(f"**Notes:** {sub.get('Auditor_Notes', 'N/A')}")
                                    st.markdown(f"**Rationale:** {sub.get('Rationale', 'N/A')}")
                                    st.markdown("---")
elif analyze_btn and not doc_text:
    st.warning("Please upload a file or paste text.")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="legal-disclaimer">
    ! LegalAIze is not official auditing software for EU AI Act compliance. 
  The assessments & interpretations made with LegalAIze, including the results presented, 
    are not to be interpreted in a legally binding context of the EU AI Act & ISO 42001:2023 Standard. 
</div>
""", unsafe_allow_html=True)