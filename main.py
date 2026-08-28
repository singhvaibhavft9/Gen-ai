import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel

# OpenAI-related code is intentionally disabled because this service uses Gemini only.
# from openai import OpenAI

load_dotenv()

# Read Gemini API key from environment. OpenAI key support is intentionally disabled.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or GEMINI_API_KEY

# Initialize FastAPI app
app = FastAPI(title="Summarizer API", version="1.0")

# Initialize the Gemini client
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is missing")

client = genai.Client(api_key=GEMINI_API_KEY)

# Request and response models
class SummaryRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150

class SummaryResponse(BaseModel):
    summary: str

@app.get("/")
def root():
    return {
        "message": "Summarizer API is running.",
        "docs": "/docs",
        "summarize": "/summarize"
    }

@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummaryRequest):
    try:
        max_length = request.max_length or 150
        if max_length <= 0:
            raise HTTPException(status_code=400, detail="max_length must be a positive integer")

        prompt = (
            f"Summarize the following text in exactly {max_length} words. "
            f"Do not exceed {max_length} words.\n\nText:\n{request.text}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        summary = (response.text or "").strip()
        summary_words = summary.split()

        if len(summary_words) > max_length:
            summary = " ".join(summary_words[:max_length])

        return {"summary": summary}
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")