import os
import re
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from tinyfish import TinyFish

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

groq_key = os.getenv("GROQ_API_KEY")
tinyfish_key = os.getenv("TINYFISH_API_KEY")

tf_client = TinyFish(api_key=tinyfish_key)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=groq_key
)

class MessageRequest(BaseModel):
    message: str

def clean_markdown_formatting(text: str) -> str:
    """Removes unwanted markdown formatting symbols for plain text rendering."""
    # Remove asterisks used for bold/italic (* or **)
    text = re.sub(r'\*+', '', text)
    # Remove header hashtags (# or ##)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        today_date = datetime.now().strftime("%B %d, %Y")
        
        # Retrieve live web search context
        try:
            search_resp = tf_client.search.query(query=request.message)
            results = search_resp.results if hasattr(search_resp, "results") else []
            search_context = "\n".join([f"- {r.title}: {r.snippet}" for r in results[:5]])
        except Exception as search_err:
            search_context = f"Search error: {str(search_err)}"

        if not search_context.strip():
            search_context = "No live search results available."

        # System prompt instructions for natural, plain text output
        prompt = (
            f"SYSTEM: You are PRIME AI, a direct assistant operating on {today_date}.\n"
            f"USER QUERY: {request.message}\n\n"
            f"LIVE SEARCH CONTEXT:\n{search_context}\n\n"
            "FORMATTING & ACCURACY RULES:\n"
            "1. Answer strictly using verified facts supported by the live search context.\n"
            "2. Write in clean, natural conversational text without markdown symbols.\n"
            "3. Do NOT use asterisks (*), hashtags (#), or bullet points with symbols.\n"
            "4. Organize ideas using clear spacing and natural sentences.\n"
            "5. Never mention knowledge cutoffs or context window limitations."
        )

        response = llm.invoke(prompt)
        raw_text = response.content
        
        # Remove leftover markdown characters before returning payload
        cleaned_reply = clean_markdown_formatting(raw_text)

        return {"reply": cleaned_reply}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/")
async def read_index():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "PRIME AI API active."}
