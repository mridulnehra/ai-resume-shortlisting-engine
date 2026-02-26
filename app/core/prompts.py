from langchain_core.prompts import PromptTemplate

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

evaluation_prompt = PromptTemplate(
    template=EVALUATION_TEMPLATE,
    input_variables=["job_description", "resume_data", "semantic_context"],
    partial_variables={"format_instructions": ""}
)

PARSING_TEMPLATE = """You are an expert data extraction AI. Your task is to extract structured information from the following raw resume text.

**Raw Resume Text:**
{raw_resume_text}

Extract the candidate's education (including inferring the tier of the institution if possible, e.g., Tier 1, Tier 2, Tier 3), their work experience (specifically looking for quantified impact/achievements and evidence of leadership/ownership), and a flat list of all their skills.

Return your response strictly adhering to the following JSON structure. DO NOT wrap it in Markdown formatting blocks like ```json :
{format_instructions}"""

parsing_prompt = PromptTemplate(
    template=PARSING_TEMPLATE,
    input_variables=["raw_resume_text"],
    partial_variables={"format_instructions": ""}
)
