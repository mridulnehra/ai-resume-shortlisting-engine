from pydantic import BaseModel, Field
from typing import List, Optional

class Education(BaseModel):
    degree: str = Field(description="The degree obtained, e.g., B.Tech, MS, PhD")
    institution: str = Field(description="Name of the institution")
    tier: str = Field(description="Tier of the institution (e.g., Tier 1, Tier 2, Tier 3) based on general reputation")
    graduation_year: Optional[str] = Field(None, description="Year of graduation")

class Experience(BaseModel):
    company: str = Field(description="Name of the company")
    role: str = Field(description="Job title/role")
    duration: Optional[str] = Field(None, description="Duration of employment")
    impact_achievements: List[str] = Field(description="Key quantified achievements or impact, e.g., 'Reduced latency by 20%'")
    leadership_ownership: List[str] = Field(description="Examples of leading projects or teams, ownership of features")

class ResumeData(BaseModel):
    candidate_name: Optional[str] = Field(None, description="Name of the candidate")
    education: List[Education] = Field(description="Educational background")
    experience: List[Experience] = Field(description="Work experience")
    skills: List[str] = Field(description="Technical and soft skills extracted from the resume")

class ScoreDetail(BaseModel):
    score: int = Field(description="Score from 0-100")
    explanation: str = Field(description="Detailed explanation of why this score was given")

class EvaluationOutput(BaseModel):
    exact_match: ScoreDetail = Field(description="Keyword overlap between JD and Resume skills/requirements")
    semantic_similarity: ScoreDetail = Field(description="Conceptual match between candidate experience/skills and the JD role")
    impact: ScoreDetail = Field(description="Rating of the quantified results and achievements in the resume")
    ownership: ScoreDetail = Field(description="Rating of the candidate's leadership, ownership, and independent contributions")
    overall_score: int = Field(description="Weighted average or combined score of the 4 dimensions (0-100)")
    tier: str = Field(description="Classification into 'Tier A' (Fast-track), 'Tier B' (Technical Screen), or 'Tier C' (Needs Evaluation) based on the overall profile")
    final_recommendation: str = Field(description="A short summary of the candidate's fit for the role and next steps.")

class EvaluationRequest(BaseModel):
    job_description: str = Field(..., description="The full text of the job description")
    resume_text: str = Field(..., description="The parsed text of the candidate's resume (or Base64 of PDF, but text is preferred for this endpoint)")
