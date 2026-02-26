from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from app.models.schemas import EvaluationOutput, ResumeData
from app.services.evaluator import parse_resume_text, evaluate_candidate
import PyPDF2
import io
import os
from dotenv import load_dotenv

load_dotenv() # Load variables like GOOGLE_API_KEY from .env

app = FastAPI(
    title="AI Resume Shortlisting Engine",
    description="Evaluates resumes against Job Descriptions using Gemini 1.5 Pro and LangChain.",
    version="1.0.0"
)

@app.post("/evaluate", response_model=EvaluationOutput)
async def evaluate_endpoint(
    job_description: str = Form(..., description="The Job Description text"),
    resume_pdf: UploadFile = File(..., description="The candidate's Resume in PDF format")
):
    """
    Endpoint to evaluate a candidate. 
    Accepts a Job Description string and a PDF Resume upload.
    Returns a structured evaluation with 4-dimensional scores and tiering.
    """
    if not resume_pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # 1. Extract text from PDF
        pdf_content = await resume_pdf.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text() + "\n"
            
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the provided PDF.")
            
        # 2. Parse raw resume text into structured JSON (ResumeData)
        structured_resume: ResumeData = parse_resume_text(resume_text)
        
        # 3. Evaluate the structured resume against the JD
        evaluation: EvaluationOutput = evaluate_candidate(
            jd_text=job_description, 
            resume_data=structured_resume,
            raw_resume_text=resume_text
        )
        
        return evaluation
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
