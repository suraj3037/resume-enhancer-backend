import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

# Load environment variables from .env file
load_dotenv()

# Ensure the Groq API key is present
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing! Please add it to your .env file.")

# Initialize Groq client
client = Groq(api_key=API_KEY)

# Define and create the data directory in the root folder if it doesn't exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Resume Enhancer API",
    description="Backend API to analyze and enhance PDF resumes using Groq LLM.",
    version="1.0.0"
)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts all text from a saved PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read PDF file: {str(e)}"
        )

def get_resume_feedback(resume_text: str) -> str:
    """Sends the extracted text to Groq LLM for analysis."""
    prompt = f"""
    You are an expert Executive Resume Writer and Technical Recruiter. 
    Analyze the following resume text and provide actionable, highly specific improvements.
    
    Structure your feedback into these sections:
    1. Overall Impression: A brief summary of the resume's current impact.
    2. Key Strengths: What the candidate did well.
    3. Actionable Improvements: Bullet points detailing specific bullet points to rewrite, weak verbs to replace with strong action verbs, and formatting/clarity suggestions.
    4. Missing Keywords & Skills: Suggest industry-standard keywords or sections that seem missing based on their profile.
    
    Here is the Resume Text:
    ---
    {resume_text}
    ---
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional resume reviewer. Give direct, constructive, and highly practical feedback."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Groq API error: {str(e)}"
        )

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "active", "service": "Resume Enhancer Backend"}

@app.post("/api/v1/enhance-resume")
async def enhance_resume(file: UploadFile = File(...)):
    """
    Upload a resume in PDF format. The file is saved locally in the 'data/' folder,
    parsed, and sent to Groq LLM for enhancement suggestions.
    """
    # 1. Validate file extension and content type
    if not file.filename.lower().endswith(".pdf") or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are accepted."
        )
    
    # 2. Save file locally inside the 'data' directory
    file_path = DATA_DIR / file.filename
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file locally: {str(e)}"
        )
    finally:
        await file.close()

    # 3. Extract text from the saved PDF
    resume_text = extract_text_from_pdf(file_path)
    if not resume_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract text from the PDF. Ensure it is not a scanned image without OCR."
        )

    # 4. Query Groq LLM for analysis
    feedback = get_resume_feedback(resume_text)

    # 5. Return structured JSON response
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "filename": file.filename,
            "saved_path": str(file_path),
            "extracted_characters": len(resume_text),
            "ai_suggestions": feedback
        }
    )