import streamlit as st
import json
import os
import time
import io

import PyPDF2
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ──────────────────────────────────────────────
# 1. Pydantic Models (from app/models/schemas.py)
# ──────────────────────────────────────────────
class Education(BaseModel):
    degree: str = Field(description="The degree obtained, e.g., B.Tech, MS, PhD")
    institution: str = Field(description="Name of the institution")
    tier: str = Field(description="Tier of the institution (e.g., Tier 1, Tier 2, Tier 3) based on general reputation")
    graduation_year: Optional[str] = Field(None, description="Year of graduation")

class Experience(BaseModel):
    company: str = Field(description="Name of the company")
    role: str = Field(description="Job title/role")
    duration: Optional[str] = Field(None, description="Duration of employment")
    impact_achievements: List[str] = Field(description="Key quantified achievements or impact")
    leadership_ownership: List[str] = Field(description="Examples of leading projects or teams")

class ResumeData(BaseModel):
    candidate_name: Optional[str] = Field(None, description="Name of the candidate")
    education: List[Education] = Field(description="Educational background")
    experience: List[Experience] = Field(description="Work experience")
    skills: List[str] = Field(description="Technical and soft skills extracted from the resume")

class ScoreDetail(BaseModel):
    score: int = Field(description="Score from 0-100")
    explanation: str = Field(description="Detailed explanation of why this score was given")

class EvaluationOutput(BaseModel):
    exact_match: ScoreDetail = Field(description="Keyword overlap between JD and Resume")
    semantic_similarity: ScoreDetail = Field(description="Conceptual match between candidate and JD")
    impact: ScoreDetail = Field(description="Rating of quantified results and achievements")
    ownership: ScoreDetail = Field(description="Rating of leadership, ownership, and independent contributions")
    overall_score: int = Field(description="Combined score of the 4 dimensions (0-100)")
    tier: str = Field(description="'Tier A' (Fast-track), 'Tier B' (Technical Screen), or 'Tier C' (Needs Evaluation)")
    final_recommendation: str = Field(description="Short summary of candidate fit and next steps")

# ──────────────────────────────────────────────
# 2. Prompt Templates (from app/core/prompts.py)
# ──────────────────────────────────────────────
EVALUATION_TEMPLATE = """You are an expert technical recruiter and AI evaluation engine. Your task is to evaluate a candidate's resume against a specific Job Description (JD) and provide a multi-dimensional score with granular explainability.

**Job Description:**
{job_description}

**Parsed Resume Data (Structured):**
{resume_data}

**Vector Database Semantic Matches (Context from ChromaDB):**
The following are chunks from the Resume that strongly match semantic requirements from the JD:
{semantic_context}

**Evaluation Criteria & Instructions:**

1. **Exact Match (0-100):** Calculate the direct keyword overlap of required skills, tools, and technologies between the JD and the Resume. Give a score and explain exactly which key terms matched and which were missing.
2. **Semantic Similarity (0-100):** Evaluate the conceptual alignment using the provided 'Vector Database Semantic Matches' as evidence, as well as by analyzing the parsed resume text yourself. For example, if the JD asks for 'Kafka' and the resume has 'AWS Kinesis' or 'GCP PubSub', this should score high. Explain the semantic overlaps found.
3. **Impact/Achievements (0-100):** Evaluate the candidate's quantifiable impact. High scores belong to statements like 'Reduced latency by 20%'. Low scores belong to vague statements like 'Worked on the backend'. Explain the assessment of their impact.
4. **Ownership (0-100):** Evaluate the candidate's level of leadership, autonomy, and ownership. Look for keywords indicating they led a team, architected a solution from scratch, or owned a critical feature. Explain the evidence (or lack thereof).

**Tiering Logic:**
Based on the four scores above, classify the candidate deeply into one of the following tiers:
- **Tier A (Fast-track):** Exceptional match across all dimensions (typically all scores > 80, highly quantifiable impact, clear ownership).
- **Tier B (Technical Screen):** Good match. Shows potential but might lack direct Exact Match or has lower ownership scores. Worth evaluating technically.
- **Tier C (Needs Evaluation/Reject):** Weak match. Missing core semantic skills, no quantifiable impact, or no overlapping basic requirements.

Return your response strictly adhering to the following JSON structure. DO NOT wrap it in Markdown formatting blocks like ```json :
{format_instructions}"""

PARSING_TEMPLATE = """You are an expert data extraction AI. Your task is to extract structured information from the following raw resume text.

**Raw Resume Text:**
{raw_resume_text}

Extract the candidate's education (including inferring the tier of the institution if possible, e.g., Tier 1, Tier 2, Tier 3), their work experience (specifically looking for quantified impact/achievements and evidence of leadership/ownership), and a flat list of all their skills.

Return your response strictly adhering to the following JSON structure. DO NOT wrap it in Markdown formatting blocks like ```json :
{format_instructions}"""

evaluation_prompt = PromptTemplate(
    template=EVALUATION_TEMPLATE,
    input_variables=["job_description", "resume_data", "semantic_context"],
    partial_variables={"format_instructions": ""}
)

parsing_prompt = PromptTemplate(
    template=PARSING_TEMPLATE,
    input_variables=["raw_resume_text"],
    partial_variables={"format_instructions": ""}
)

# ──────────────────────────────────────────────
# 3. Semantic Matcher (from app/services/chroma_service.py)
# ──────────────────────────────────────────────
class SemanticMatcher:
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def compute_semantic_match(self, jd_text: str, resume_text: str) -> str:
        """Calculates semantic match using ChromaDB vector search."""
        try:
            jd_chunks = self.text_splitter.split_text(jd_text)
            resume_chunks = self.text_splitter.split_text(resume_text)

            if not jd_chunks or not resume_chunks:
                return "Insufficient text for mapping."

            vectorstore = Chroma.from_texts(
                texts=resume_chunks,
                embedding=self.embedding_function
            )

            match_results = []
            for jd_chunk in jd_chunks:
                docs = vectorstore.similarity_search_with_score(jd_chunk, k=2)
                for doc, score in docs:
                    if score < 1.5:
                        match_results.append(
                            f"JD Requirement: '{jd_chunk[:100]}...' -> "
                            f"Candidate Evidence: '{doc.page_content}' (Distance: {score:.2f})"
                        )

            if not match_results:
                return "No strong semantic matches found via vector search."

            return "\n".join(match_results[:5])
        except Exception as e:
            return f"Error during semantic matching: {str(e)}"

# ──────────────────────────────────────────────
# 4. Evaluator Logic (from app/services/evaluator.py)
# ──────────────────────────────────────────────
def _invoke_with_retry(chain, inputs, status_placeholder=None, max_retries=3):
    """Invoke a LangChain chain with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 25 * (attempt + 1)
                if status_placeholder:
                    status_placeholder.warning(f"⏳ Rate limited. Retrying in {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise e
    raise ValueError("Max retries exceeded due to rate limiting. Please wait a minute and try again.")

# ──────────────────────────────────────────────
# 5. Initialize heavy resources (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def get_semantic_matcher():
    """Cache the embedding model so it only loads once."""
    return SemanticMatcher()

def get_llm(api_key: str):
    """Create an LLM instance with the provided API key."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        max_retries=3,
        google_api_key=api_key,
    )

# ──────────────────────────────────────────────
# 6. Page Config & Custom CSS
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Shortlister",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 7. Helper Functions
# ──────────────────────────────────────────────
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
# 8. Header
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">🎯 AI Resume Shortlister</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Multi-dimensional resume evaluation powered by Gemini 2.5 Flash & ChromaDB</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 9. Sidebar — Inputs
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 API Key")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        help="Get your key at https://aistudio.google.com/apikey",
        value=os.environ.get("GOOGLE_API_KEY", "")
    )

    st.markdown("---")

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
        height=200,
        placeholder="e.g. Looking for a Software Engineer with 2+ years experience in Python, FastAPI, and Kafka...",
    )

    st.markdown("---")
    evaluate_btn = st.button("🚀 Evaluate Candidate", use_container_width=True)

# ──────────────────────────────────────────────
# 10. Main Content — Evaluation Logic & Results
# ──────────────────────────────────────────────
if evaluate_btn:
    if not api_key:
        st.error("⚠️ Please enter your Google Gemini API key in the sidebar.")
    elif not uploaded_file:
        st.error("⚠️ Please upload a resume PDF.")
    elif not job_description.strip():
        st.error("⚠️ Please paste a job description.")
    else:
        status = st.empty()
        progress = st.progress(0, text="Starting evaluation...")

        try:
            # Step 1: Extract text from PDF
            status.info("📄 Step 1/4: Extracting text from PDF...")
            progress.progress(10, text="Extracting PDF text...")
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text() + "\n"

            if not resume_text.strip():
                st.error("❌ Could not extract text from the PDF. Please try a different file.")
                st.stop()

            progress.progress(20, text="PDF text extracted!")

            # Step 2: Semantic matching with ChromaDB
            status.info("🧠 Step 2/4: Running semantic matching (ChromaDB)...")
            progress.progress(30, text="Computing semantic embeddings...")
            matcher = get_semantic_matcher()
            semantic_context = matcher.compute_semantic_match(job_description, resume_text)
            progress.progress(40, text="Semantic matching complete!")

            # Step 3: LLM parsing
            status.info("🤖 Step 3/4: Parsing resume with Gemini 2.5 Flash...")
            progress.progress(50, text="LLM parsing resume...")
            llm = get_llm(api_key)
            resume_parser = PydanticOutputParser(pydantic_object=ResumeData)
            _parse_prompt = parsing_prompt.partial(format_instructions=resume_parser.get_format_instructions())
            parse_chain = _parse_prompt | llm | resume_parser
            structured_resume = _invoke_with_retry(parse_chain, {"raw_resume_text": resume_text}, status)
            progress.progress(70, text="Resume parsed successfully!")

            # Brief pause to avoid rate limiting
            time.sleep(3)

            # Step 4: LLM evaluation
            status.info("📊 Step 4/4: Evaluating candidate against JD...")
            progress.progress(80, text="LLM evaluating candidate...")
            evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationOutput)
            _eval_prompt = evaluation_prompt.partial(format_instructions=evaluation_parser.get_format_instructions())
            eval_chain = _eval_prompt | llm | evaluation_parser
            resume_data_str = structured_resume.model_dump_json(indent=2)
            result = _invoke_with_retry(eval_chain, {
                "job_description": job_description,
                "resume_data": resume_data_str,
                "semantic_context": semantic_context
            }, status)

            progress.progress(100, text="✅ Evaluation complete!")
            status.success("✅ Evaluation complete!")
            time.sleep(0.5)
            status.empty()
            progress.empty()

            # ── Display Results ──

            # Overall Score + Tier
            col_overall, col_tier = st.columns([1, 1])
            with col_overall:
                st.markdown(f"""
                <div class="overall-score-container">
                    <div class="score-label">Overall Score</div>
                    <div class="overall-score-value">{result.overall_score}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_tier:
                tier_class = get_tier_class(result.tier)
                st.markdown(f"""
                <div class="overall-score-container">
                    <div class="score-label">Candidate Tier</div>
                    <div style="margin-top: 0.5rem;">
                        <span class="tier-badge {tier_class}">{result.tier}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # 4 Score Cards
            st.markdown("### 📊 Multi-Dimensional Scores")
            c1, c2 = st.columns(2)
            with c1:
                render_score_card("Exact Match", result.exact_match.score, result.exact_match.explanation, "🔑")
            with c2:
                render_score_card("Semantic Similarity", result.semantic_similarity.score, result.semantic_similarity.explanation, "🧠")

            c3, c4 = st.columns(2)
            with c3:
                render_score_card("Impact & Achievements", result.impact.score, result.impact.explanation, "📈")
            with c4:
                render_score_card("Ownership & Leadership", result.ownership.score, result.ownership.explanation, "👑")

            st.markdown("---")

            # Recommendation
            st.markdown("### 💡 Final Recommendation")
            st.markdown(f"""
            <div class="recommendation-box">
                {result.final_recommendation}
            </div>
            """, unsafe_allow_html=True)

            # Parsed Resume Data (collapsible)
            with st.expander("📋 View Parsed Resume Data"):
                st.json(json.loads(structured_resume.model_dump_json(indent=2)))

            # Raw JSON (collapsible)
            with st.expander("🔧 View Raw Evaluation JSON"):
                st.json(json.loads(result.model_dump_json(indent=2)))

        except Exception as e:
            progress.empty()
            status.empty()
            st.error(f"❌ Error: {str(e)}")

else:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="score-card">
            <div class="score-label">📄 Step 1</div>
            <div style="color: #c8cbd9; font-size: 0.95rem;">Enter your Gemini API key & upload a resume PDF</div>
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
