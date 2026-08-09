import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "openai/gpt-oss-120b"

app = FastAPI()

# ---------------------------------------------------------------------------
# CORS — required so the frontend (a different origin) can call this API.
# Replace "*" with your deployed frontend URL once you know it, e.g.
# ["https://your-portfolio.vercel.app"], for a tighter setup.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resume PDF lives next to this file so it ships with the deployment bundle.
RESUME_PATH = Path(__file__).parent / "resume.pdf"


class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []


resume_schema = Resume.model_json_schema()


class ChatRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# The resume rarely changes, so parse it once per server instance and reuse
# it for every /chat call instead of re-reading the PDF and calling the LLM
# to re-parse it on every single question. This cuts latency and Groq usage
# a lot, which matters on a serverless platform that bills/limits per call.
# ---------------------------------------------------------------------------
_resume_cache: Resume | None = None


def get_resume() -> Resume:
    global _resume_cache
    if _resume_cache is None:
        if not RESUME_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Resume file not found at {RESUME_PATH}. "
                f"Make sure resume.pdf is deployed alongside index.py.",
            )
        resume_text = read_pdf(RESUME_PATH)
        _resume_cache = parse_resume(resume_text)
    return _resume_cache


def ask_candidate(question: str, resume: Resume):
    system_prompt = f"""
    You are Raunak Jha, answering interview questions live, in first person — as if you're 
    actually speaking to the interviewer, not reading from a document.

    Here is everything you know about your own background:

    {resume.model_dump_json(indent=2)}

    Rules:
    1. Answer using only the information above. Never hallucinate or invent details.
    2. If the information isn't available, say "I don't have enough information to answer that."
    3. Speak naturally and conversationally — short, direct sentences, like real speech.
    4. Do NOT use markdown tables, pipes (|), hyphens as bullets, or asterisks for bold.
       Plain text only, no formatting symbols.
    5. Do NOT summarize your entire resume, education, or work history unless the question 
       directly asks for it.
    6. Synthesize and rephrase in your own words — never copy resume bullet points verbatim.
    7. Keep answers focused: 3-5 sentences unless the question genuinely needs more depth.
    """

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content


def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format = {"type": "json_object"}
    response = client.chat.completions.create(
        model=model, messages=messages, response_format=response_format
    )
    raw_output = response.choices[0].message.content
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


@app.get("/")
def home():
    return {"message": "Resume bot is running"}


@app.post("/chat")
def chat(request: ChatRequest):
    resume = get_resume()
    answer = ask_candidate(request.question, resume)
    return {"answer": answer}