import streamlit as st
import requests
import json
import os
import time
from fpdf import FPDF
from pypdf import PdfReader

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
    # (Assuming structure: LegalAIze -> frontend -> pages -> ThisScript.py)
    project_root = os.path.abspath(os.path.join(current_script_dir, "..", ".."))
    
    # 3. Build path to data folder
    MAPPING_FILE = os.path.join(project_root, "data", "mapping.json")
except Exception as e:
    # Fallback just in case
    MAPPING_FILE = "mapping.json"
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

@st.cache_data
def load_mapping_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"JSON Error: {e}")
        return {}

MAPPING_DATA = load_mapping_data(MAPPING_FILE)

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
        color: #6b7280;
        text-align: center;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
        font-style: italic;
    }
    
    /* TOP SPACING */
    .block-container { padding-top: 2rem; }

    /* --- BUTTON STYLING (Matching Home Page Blue) --- */
    div.stButton > button:first-child, 
    div.stDownloadButton > button:first-child {
        background-color: #1e3a8a !important; /* Deep Navy Blue */
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
    }

    div.stButton > button:first-child:hover,
    div.stDownloadButton > button:first-child:hover {
        background-color: #172554 !important; /* Darker Navy */
        color: #e2e8f0 !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* TEXT HIGHLIGHTS */
    h1, h2, h3 {
        color: #1e3a8a !important;
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

    # --- REQUIREMENTS ---
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, "Detailed Requirements Analysis", ln=True)
    pdf.ln(5)

    for req in requirements:
        def clean_text(text):
            if not text: return ""
            txt_fixed = text.replace("—", "-").replace("–", "-").replace("’", "'").replace('“', '"').replace('”', '"')
            return str(txt_fixed).encode('latin-1', 'replace').decode('latin-1')

        name = clean_text(req['name'])
        notes = clean_text(req['notes'])
        
        mapping_info = MAPPING_DATA.get(req['name'], {})
        iso_ref = clean_text(mapping_info.get("iso_ref", "N/A"))
        
        ai_articles = mapping_info.get("ai_act_articles", [])
        ai_refs_str = ", ".join([a.get("ref", "") for a in ai_articles])
        if not ai_refs_str: ai_refs_str = "N/A"
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

    return pdf.output(dest='S').encode('latin-1', 'replace')

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### 📋 System Status")
    try:
        if requests.get(f"{BACKEND_URL}/health", timeout=1).status_code == 200:
            st.success("System Online")
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
st.title("🔍 Audit Compliance")
st.markdown("Automated analysis of technical documentation powered by AI.")

if not MAPPING_DATA:
    st.warning(f"⚠️ Warning: 'mapping.json' not found at expected path: {MAPPING_FILE}")

# 1. INPUT SECTION
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
            # on_change=reset_session -> Clears analysis when file changes
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
                headers=headers, timeout=120
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
                    r_id = item.get("Mapped_ID", "N/A")
                    r_notes = item.get("Auditor_Notes", "No notes available.")
                    score = item.get("Score", 0) 
                    
                    processed_reqs.append({
                        "name": r_name, "id": r_id, "score_display": f"{score}/5", 
                        "progress": score / 5.0, "notes": r_notes
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

        # --- PDF DOWNLOAD ---
        c_pdf_left, c_pdf_mid, c_pdf_right = st.columns([1, 2, 1])
        with c_pdf_mid:
            pdf_bytes = create_pdf_report(
                results, glob_score, 
                st.session_state.total_points, st.session_state.max_points
            )
            st.download_button(
                label="📄 DOWNLOAD FULL PDF REPORT", 
                data=pdf_bytes,
                file_name="compliance_report.pdf",
                mime="application/pdf",
                type="primary", 
                use_container_width=True,
                icon="📥"
            )

        st.markdown("---")
        
        # --- REQUIREMENTS DETAIL ---
        st.subheader("📋 Requirements Detail")
        
        results_sorted = sorted(results, key=lambda x: x["progress"])

        for req in results_sorted:
            if req["progress"] >= 0.8:
                  icon = "✅"
            elif req["progress"] >= 0.4:
                  icon = "⚠️"
            else:
                  icon = "❌"
            
            # Retrieve Mapping Info
            map_info = MAPPING_DATA.get(req['name'], {})
            iso_ui = map_info.get("iso_ref", "N/A")
            ai_act_list = map_info.get("ai_act_articles", [])
            ai_act_ui = ", ".join([item.get("ref", "") for item in ai_act_list])
            if not ai_act_ui: ai_act_ui = "N/A"
            
            with st.expander(f"{icon} {req['name']} ({req['score_display']})"):
                col_A, col_B = st.columns([1, 2])
                with col_A:
                    st.caption("DETAILS")
                    st.markdown(f"**ID:** `{req['id']}`")
                    st.markdown(f"**Score:** {req['score_display']}")
                    
                    st.caption("REFERENCES")
                    st.markdown(f"**ISO:** {iso_ui}")
                    st.markdown(f"**AI Act:** {ai_act_ui}")
                    
                    st.progress(req["progress"]) 
                with col_B:
                    st.caption("AI FINDINGS")
                    st.write(req["notes"])

elif analyze_btn and not doc_text:
    st.warning("Please upload a file or paste text.")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="legal-disclaimer">
    ! LegalAIze is not official auditing software for EU AI Act compliance. 
    The assessments & interpretations made with LegalAIze, including the results presented, 
    are not to be interpreted in a legally binding context of the EU AI Act & ISO Standard. 
</div>
""", unsafe_allow_html=True)