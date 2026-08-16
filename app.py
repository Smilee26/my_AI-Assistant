import os
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

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        today_date = datetime.now().strftime("%B %d, %Y")
        
        # Fetch up to 5 web results for higher detail
        try:
            search_resp = tf_client.search.query(query=request.message)
            results = search_resp.results if hasattr(search_resp, "results") else []
            search_context = "\n".join([f"- {r.title}: {r.snippet}" for r in results[:5]])
        except Exception as search_err:
            search_context = f"Search error: {str(search_err)}"

        if not search_context.strip():
            search_context = "No search results retrieved."

        # Expanded system prompt for synthesized, complete answers
        prompt = (
            f"SYSTEM: You are PRIME AI operating on {today_date}.\n"
            f"USER QUERY: {request.message}\n\n"
            f"LIVE SEARCH RESULTS:\n{search_context}\n\n"
            "RULES:\n"
            "1. Use live search data as your primary source for up-to-date real-time context.\n"
            "2. Synthesize all retrieved findings into a comprehensive, direct, and detailed answer.\n"
            "3. Do NOT mention knowledge cutoffs."
        )

        response = llm.invoke(prompt)
        return {"reply": response.content}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/")
async def read_index():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "PRIME AI API active."}
