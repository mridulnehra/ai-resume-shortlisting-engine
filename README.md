# 🎯 AI Resume Shortlisting & Evaluation Engine

An AI-powered recruitment system that evaluates resumes against Job Descriptions using multi-dimensional scoring, semantic matching, and explainable AI.

**Built for the Internship Take-Home Assignment — Option A: Evaluation & Scoring Engine (Depth over Breadth)**

---

## 🏗️ Architecture Overview

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Streamlit   │────▶│   FastAPI API     │────▶│  Gemini 2.5 Flash│
│  Frontend    │◀────│   /evaluate       │◀────│  (via LangChain) │
│  :8501       │     │   :8000           │     └──────────────────┘
└─────────────┘     └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │    ChromaDB       │
                    │  (Embeddings +   │
                    │   Vector Search) │
                    └──────────────────┘
```

## 🧠 How It Works

1. **PDF Ingestion** → PyPDF2 extracts raw text from the resume
2. **LLM Parsing** → Gemini 2.5 Flash transforms raw text into structured JSON (Education with Tiering, Experience with Impact, Skills)
3. **Semantic Matching** → ChromaDB embeds resume chunks and finds conceptual overlaps with the JD (e.g., "AWS Kinesis" ≈ "Kafka")
4. **Multi-Dimensional Scoring** → Gemini evaluates the candidate on 4 axes (0–100 each):
   - **Exact Match**: Direct keyword overlap
   - **Semantic Similarity**: Conceptual alignment via embeddings
   - **Impact/Achievements**: Quantified results (e.g., "Reduced latency by 20%")
   - **Ownership**: Leadership and autonomy evidence
5. **Explainability** → Every score includes a "Why" explanation
6. **Tiering** → Candidates classified into Tier A (Fast-track), Tier B (Technical Screen), or Tier C (Needs Evaluation)

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Backend API | FastAPI + Uvicorn |
| LLM | Google Gemini 2.5 Flash |
| Reasoning Layer | LangChain |
| Vector Store | ChromaDB (in-memory) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| PDF Parsing | PyPDF2 |
| Data Validation | Pydantic v2 |
| Frontend UI | Streamlit |

## 📁 Project Structure

```
resume-shortlisting-app/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app + /evaluate endpoint
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic models (ResumeData, EvaluationOutput)
│   ├── core/
│   │   ├── __init__.py
│   │   └── prompts.py             # LangChain prompt templates
│   ├── services/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # LLM parsing + scoring orchestration
│   │   └── chroma_service.py      # ChromaDB semantic matching
│   └── api/
│       └── __init__.py
├── streamlit_app.py               # Streamlit frontend UI
├── requirements.txt
├── SYSTEM_DESIGN.md               # Architecture & design document
├── .env                           # API keys (not committed)
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- A Google Gemini API key ([Get one here](https://aistudio.google.com/apikey))

### 1. Clone & Install

```bash
git clone <repo-url>
cd resume-shortlisting-app

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Start the Backend

```bash
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be live at http://localhost:8000. Swagger docs at http://localhost:8000/docs.

### 4. Start the Frontend (Streamlit UI)

In a **separate terminal**:

```bash
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser.

### 5. Test It

1. Upload a PDF resume via the sidebar
2. Paste a job description
3. Click **🚀 Evaluate Candidate**
4. View scores, tier classification, and explanations

## 📖 API Reference

### `POST /evaluate`

**Content-Type:** `multipart/form-data`

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_description` | string (form field) | The full job description text |
| `resume_pdf` | file (PDF) | The candidate's resume |

**Response (200 OK):**

```json
{
  "exact_match": { "score": 85, "explanation": "Candidate has Python, FastAPI..." },
  "semantic_similarity": { "score": 78, "explanation": "Strong conceptual match..." },
  "impact": { "score": 90, "explanation": "Reduced latency by 20%..." },
  "ownership": { "score": 75, "explanation": "Led development of analytics..." },
  "overall_score": 82,
  "tier": "Tier A",
  "final_recommendation": "Fast-track. Strong technical fit."
}
```

### `GET /health`

Returns `{"status": "healthy"}` - used for monitoring and load balancer health checks.

## 🔮 Scalability Considerations (10,000 resumes/day)

- **Async Processing**: FastAPI is async-native; can be paired with Celery/Redis for background job queues.
- **Vector Store**: ChromaDB can be swapped with Pinecone or Weaviate for production-grade persistent embeddings.
- **Caching**: Parsed resume structures can be cached to avoid re-parsing the same PDF.
- **Rate Limiting**: Built-in retry logic with exponential backoff handles Gemini API rate limits.
- **Horizontal Scaling**: Stateless FastAPI instances can be scaled behind a load balancer.

## 📝 Author

Mridul Nehra — Data Science Student
