import json
import os
import time
from dotenv import load_dotenv

# Load .env BEFORE any Google API calls
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from app.models.schemas import EvaluationOutput, ResumeData
from app.core.prompts import evaluation_prompt, parsing_prompt
from app.services.chroma_service import semantic_matcher

# Now GOOGLE_API_KEY is available in environment
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_retries=3,  # Retry up to 3 times on transient errors
)

resume_parser = PydanticOutputParser(pydantic_object=ResumeData)
evaluation_parser = PydanticOutputParser(pydantic_object=EvaluationOutput)

def _invoke_with_retry(chain, inputs, max_retries=3):
    """Invoke a LangChain chain with automatic retry on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 25 * (attempt + 1)  # 25s, 50s, 75s
                print(f"Rate limited (attempt {attempt+1}/{max_retries}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    raise ValueError("Max retries exceeded due to rate limiting. Please wait a minute and try again.")

def parse_resume_text(raw_text: str) -> ResumeData:
    """Parses raw resume text into a structured Pydantic model using LLM."""
    _prompt = parsing_prompt.partial(format_instructions=resume_parser.get_format_instructions())
    chain = _prompt | llm | resume_parser
    try:
        return _invoke_with_retry(chain, {"raw_resume_text": raw_text})
    except Exception as e:
        print(f"Error parsing resume: {e}")
        raise ValueError(f"Failed to parse resume text: {e}")

def evaluate_candidate(jd_text: str, resume_data: ResumeData, raw_resume_text: str) -> EvaluationOutput:
    """Evaluates the structured resume against the JD using LLM and ChromaDB context."""
    
    # Get chroma context
    semantic_context = semantic_matcher.compute_semantic_match(jd_text, raw_resume_text)
    
    _prompt = evaluation_prompt.partial(format_instructions=evaluation_parser.get_format_instructions())
    chain = _prompt | llm | evaluation_parser
    
    resume_data_str = resume_data.model_dump_json(indent=2)
    
    # Small delay between the two LLM calls to avoid back-to-back rate limits
    time.sleep(5)
    
    try:
        return _invoke_with_retry(chain, {
            "job_description": jd_text,
            "resume_data": resume_data_str,
            "semantic_context": semantic_context
        })
    except Exception as e:
        print(f"Error evaluating candidate: {e}")
        raise ValueError(f"Failed to evaluate candidate: {e}")
