# System Design Document

## 1. Overview

This document describes the architecture, data flow, and design decisions for the AI Resume Shortlisting Engine — a system that evaluates candidate resumes against Job Descriptions using multi-dimensional scoring and explainable AI.

## 2. High-Level Architecture

```
                         ┌───────────────────────────────────────┐
                         │           Streamlit Frontend           │
                         │  (Upload PDF + Paste JD + View Scores)│
                         └──────────────┬────────────────────────┘
                                        │ HTTP POST /evaluate
                         ┌──────────────▼────────────────────────┐
                         │          FastAPI Backend               │
                         │                                        │
                         │  ┌──────────────────────────────────┐ │
                         │  │   1. PDF Text Extraction (PyPDF2)│ │
                         │  └──────────────┬───────────────────┘ │
                         │                 │ raw text             │
                         │  ┌──────────────▼───────────────────┐ │
                         │  │   2. LLM Resume Parser           │ │
                         │  │   (Gemini 2.5 Flash + LangChain) │ │
                         │  │   Output: Structured ResumeData  │ │
                         │  └──────────────┬───────────────────┘ │
                         │                 │ structured JSON      │
                         │  ┌──────────────▼───────────────────┐ │
                         │  │   3. Semantic Matcher (ChromaDB)  │ │
                         │  │   Embeds resume → queries JD      │ │
                         │  │   Output: Semantic context string │ │
                         │  └──────────────┬───────────────────┘ │
                         │                 │ semantic_context     │
                         │  ┌──────────────▼───────────────────┐ │
                         │  │   4. LLM Evaluator               │ │
                         │  │   Inputs: JD + ResumeData +      │ │
                         │  │           semantic_context        │ │
                         │  │   Output: 4 Scores + Tier + Why  │ │
                         │  └──────────────────────────────────┘ │
                         └───────────────────────────────────────┘
```

## 3. Data Flow

### Input
- **Job Description**: Plain text string
- **Resume**: PDF file

### Processing Pipeline
1. **Extract** → PyPDF2 reads the PDF and outputs raw text
2. **Parse** → LLM (Gemini 2.5 Flash) converts raw text into structured `ResumeData` JSON containing:
   - Education (with institutional tier: Tier 1/2/3)
   - Experience (with impact/achievements and leadership evidence)
   - Skills (flat list)
3. **Embed & Match** → Resume text is chunked (500 chars, 50 overlap) and embedded into ChromaDB using `all-MiniLM-L6-v2`. JD chunks are queried against this store to find semantic overlaps.
4. **Score** → LLM receives structured resume + JD + semantic context and outputs:
   - 4 scores (0–100) with explanations
   - Overall score
   - Tier classification (A/B/C)
   - Final recommendation

### Output: `EvaluationOutput`
```json
{
  "exact_match":        { "score": int, "explanation": str },
  "semantic_similarity": { "score": int, "explanation": str },
  "impact":             { "score": int, "explanation": str },
  "ownership":          { "score": int, "explanation": str },
  "overall_score":      int,
  "tier":               str,
  "final_recommendation": str
}
```

## 4. Tech Stack Choices

| Choice | Rationale |
|--------|-----------|
| **FastAPI** | Async-native, auto-generates OpenAPI docs, Pydantic integration, production-grade |
| **LangChain** | Composable chains, output parsers for structured JSON, prompt template management |
| **Gemini 2.5 Flash** | Fast inference, strong instruction-following, free tier available |
| **ChromaDB** | Zero-config in-memory vector store, perfect for prototyping semantic search |
| **Sentence-Transformers** | `all-MiniLM-L6-v2` runs locally, no API calls needed, 384-dim embeddings |
| **Pydantic v2** | Runtime type validation ensures LLM outputs conform to expected schema |
| **Streamlit** | Rapid UI prototyping in Python, drag-and-drop file uploads |

## 5. Scoring Logic Deep Dive

### Exact Match
The LLM compares required skills/tools from the JD against the resume's skills list. It identifies exact keyword matches and notes missing requirements. High scores indicate strong overlap; low scores indicate missing core requirements.

### Semantic Similarity
ChromaDB embeds resume chunks and finds the nearest vectors to JD requirement chunks. This allows recognition of conceptual equivalence:
- "AWS Kinesis" ↔ "Kafka" (both are streaming platforms)
- "React" ↔ "Frontend Development"
- "PostgreSQL" ↔ "Relational Databases"

The top 5 semantic matches (with L2 distance < 1.5) are passed as evidence to the LLM.

### Impact/Achievements
The LLM evaluates quantified statements. Examples:
- **High score**: "Reduced latency by 20%", "Grew user base from 10K to 100K"
- **Low score**: "Worked on the backend", "Helped with testing"

### Ownership
The LLM checks for leadership indicators:
- "Led a team of 5 engineers"
- "Architected the microservice from scratch"
- "Owned the end-to-end feature delivery"

### Tiering Logic
| Tier | Criteria |
|------|----------|
| **Tier A (Fast-track)** | All scores > 80, clear quantified impact, strong ownership |
| **Tier B (Technical Screen)** | Good match but gaps in exact match or ownership |
| **Tier C (Needs Evaluation)** | Missing core skills, no quantified impact |

## 6. Prompt Engineering Strategy

We use a **single-pass evaluation prompt** that:
1. Receives the full JD, structured resume JSON, and ChromaDB semantic context
2. Is explicitly instructed to evaluate each of 4 dimensions independently
3. Must provide an explanation for every score (enforced by Pydantic output parser)
4. Uses `PydanticOutputParser` to guarantee JSON conformity with the `EvaluationOutput` schema

The prompt explicitly tells the LLM to NOT wrap output in markdown code blocks, which prevents common parsing failures.

## 7. Error Handling & Resilience

- **Rate Limiting**: Custom `_invoke_with_retry` function retries on 429 errors with 25s exponential backoff (up to 3 retries)
- **PDF Validation**: Rejects non-PDF uploads with 400 error
- **Empty Text Detection**: Returns 400 if PDF text extraction yields empty content
- **LLM Failure Handling**: Wraps all LLM calls in try/except, surfaces meaningful error messages

## 8. Scalability Path (Production Readiness for 10,000 Resumes/Day)

| Current | Production Upgrade |
|---------|-------------------|
| ChromaDB in-memory | Pinecone / Weaviate (persistent, distributed) |
| Synchronous processing | Celery + Redis job queue |
| Single instance | Kubernetes with HPA |
| Gemini free tier | Gemini Pro with pay-as-you-go |
| No caching | Redis cache for parsed resumes |
| No auth | OAuth2 / API key auth |
