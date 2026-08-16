import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Resolve path relative to app.py location
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Ensure the static directory exists on runtime
STATIC_DIR.mkdir(parents=True, exist_ok=True)

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
        current_year = datetime.now().year
        today_date = datetime.now().strftime("%B %d, %Y")
        
        try:
            search_resp = tf_client.search.query(query=request.message)
            results = search_resp.results if hasattr(search_resp, "results") else []
            search_context = "\n".join([f"- {r.title}: {r.snippet}" for r in results[:3]])
        except Exception as search_err:
            search_context = f"Search error: {str(search_err)}"

        if not search_context.strip():
            search_context = "No search results retrieved."

        prompt = (
            f"SYSTEM: You are PRIME AI operating on {today_date}.\n"
            f"USER QUERY: {request.message}\n\n"
            f"LIVE SEARCH RESULTS:\n{search_context}\n\n"
            "RULES:\n"
            "1. Base your response strictly on the live search data.\n"
            "2. Do NOT mention knowledge cutoff dates (e.g., 2023).\n"
            "3. Answer directly, clearly, and concisely."
        )

        response = llm.invoke(prompt)
        return {"reply": response.content}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def read_index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "PRIME AI API active. Place index.html inside the static folder."}
