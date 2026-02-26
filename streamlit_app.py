import streamlit as st
import requests
import json

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Shortlister",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS for Premium Dark Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #0f1117; }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b8fa3;
        text-align: center;
        margin-bottom: 2rem;
    }

    .score-card {
        background: linear-gradient(145deg, #1a1d2e, #1e2235);
        border: 1px solid #2a2d42;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .score-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }

    .score-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .score-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .score-high { color: #00d68f; }
    .score-medium { color: #ffaa00; }
    .score-low { color: #ff3d71; }

    .explanation-text {
        font-size: 0.85rem;
        color: #a0a3b8;
        line-height: 1.6;
        margin-top: 0.5rem;
    }

    .tier-badge {
        display: inline-block;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        letter-spacing: 1px;
    }

    .tier-a {
        background: linear-gradient(135deg, #00d68f, #00b887);
        color: #000;
    }

    .tier-b {
        background: linear-gradient(135deg, #ffaa00, #ff8800);
        color: #000;
    }

    .tier-c {
        background: linear-gradient(135deg, #ff3d71, #ff1a53);
        color: #fff;
    }

    .overall-score-container {
        background: linear-gradient(145deg, #1a1d2e, #1e2235);
        border: 2px solid #667eea;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }

    .overall-score-value {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .recommendation-box {
        background: linear-gradient(145deg, #1a1d2e, #1e2235);
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #c8cbd9;
        line-height: 1.7;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .sidebar .stTextArea textarea {
        background: #1a1d2e;
        border: 1px solid #2a2d42;
        border-radius: 12px;
        color: #c8cbd9;
    }

    div[data-testid="stFileUploader"] {
        background: #1a1d2e;
        border: 2px dashed #2a2d42;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────
FASTAPI_URL = "http://localhost:8000"

def get_score_class(score: int) -> str:
    if score >= 75: return "score-high"
    if score >= 50: return "score-medium"
    return "score-low"

def get_tier_class(tier: str) -> str:
    if "A" in tier: return "tier-a"
    if "B" in tier: return "tier-b"
    return "tier-c"

def render_score_card(label: str, score: int, explanation: str, icon: str):
    score_class = get_score_class(score)
    st.markdown(f"""
    <div class="score-card">
        <div class="score-label">{icon} {label}</div>
        <div class="score-value {score_class}">{score}</div>
        <div class="explanation-text">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">🎯 AI Resume Shortlister</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Multi-dimensional resume evaluation powered by Gemini 2.5 Flash & ChromaDB</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar — Inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader(
        "Drop a PDF resume here",
        type=["pdf"],
        help="Upload the candidate's resume in PDF format"
    )

    st.markdown("---")

    st.markdown("### 📋 Job Description")
    job_description = st.text_area(
        "Paste the JD here",
        height=250,
        placeholder="e.g. Looking for a Software Engineer with 2+ years experience in Python, FastAPI, and Kafka. Should have experience building scalable microservices and reducing system latency. Leadership experience preferred.",
        help="Enter the full job description text to evaluate against"
    )

    st.markdown("---")

    evaluate_btn = st.button("🚀 Evaluate Candidate", use_container_width=True)

# ──────────────────────────────────────────────
# Main Content — Results
# ──────────────────────────────────────────────
if evaluate_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("⚠️ Please paste a job description.")
    else:
        with st.spinner("🔍 Analyzing resume... This may take 30-60 seconds."):
            try:
                files = {"resume_pdf": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"job_description": job_description}

                response = requests.post(f"{FASTAPI_URL}/evaluate", files=files, data=data, timeout=120)

                if response.status_code == 200:
                    result = response.json()

                    # ── Overall Score + Tier ──
                    col_overall, col_tier = st.columns([1, 1])

                    with col_overall:
                        st.markdown(f"""
                        <div class="overall-score-container">
                            <div class="score-label">Overall Score</div>
                            <div class="overall-score-value">{result['overall_score']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_tier:
                        tier_class = get_tier_class(result['tier'])
                        st.markdown(f"""
                        <div class="overall-score-container">
                            <div class="score-label">Candidate Tier</div>
                            <div style="margin-top: 0.5rem;">
                                <span class="tier-badge {tier_class}">{result['tier']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

                    # ── 4 Score Cards ──
                    st.markdown("### 📊 Multi-Dimensional Scores")
                    c1, c2 = st.columns(2)
                    with c1:
                        render_score_card("Exact Match", result['exact_match']['score'], result['exact_match']['explanation'], "🔑")
                    with c2:
                        render_score_card("Semantic Similarity", result['semantic_similarity']['score'], result['semantic_similarity']['explanation'], "🧠")

                    c3, c4 = st.columns(2)
                    with c3:
                        render_score_card("Impact & Achievements", result['impact']['score'], result['impact']['explanation'], "📈")
                    with c4:
                        render_score_card("Ownership & Leadership", result['ownership']['score'], result['ownership']['explanation'], "👑")

                    st.markdown("---")

                    # ── Recommendation ──
                    st.markdown("### 💡 Final Recommendation")
                    st.markdown(f"""
                    <div class="recommendation-box">
                        {result['final_recommendation']}
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Raw JSON (collapsible) ──
                    with st.expander("🔧 View Raw JSON Response"):
                        st.json(result)

                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ Error ({response.status_code}): {error_detail}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the FastAPI backend. Make sure it's running on http://localhost:8000")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The Gemini API may be rate-limited. Please wait 30 seconds and try again.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

else:
    # ── Landing state ──
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="score-card">
            <div class="score-label">📄 Step 1</div>
            <div style="color: #c8cbd9; font-size: 0.95rem;">Upload a resume PDF in the sidebar</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="score-card">
            <div class="score-label">📋 Step 2</div>
            <div style="color: #c8cbd9; font-size: 0.95rem;">Paste the job description text</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="score-card">
            <div class="score-label">🚀 Step 3</div>
            <div style="color: #c8cbd9; font-size: 0.95rem;">Click Evaluate to get AI-powered results</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #555; font-size: 0.85rem; margin-top: 2rem;">
        Powered by <strong>Gemini 2.5 Flash</strong> · <strong>LangChain</strong> · <strong>ChromaDB</strong> · <strong>FastAPI</strong>
    </div>
    """, unsafe_allow_html=True)
