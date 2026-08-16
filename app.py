import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

search_tool = DuckDuckGoSearchRun()
groq_key = os.getenv("GROQ_API_KEY")

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
        
        # 1. Optimize search query explicitly for live data/weather
        query_text = request.message.strip()
        search_query = f"{query_text} live update {current_year}"
        
        try:
            search_results = search_tool.run(search_query)
        except Exception as search_err:
            search_results = f"Search tool failed: {str(search_err)}"

        if not search_results or len(search_results.strip()) == 0:
            search_results = "No search results found."

        # 2. Strict instruction prompt that prevents keyword misinterpretation
        prompt = (
            f"SYSTEM ROLE: You are PRIME AI, a real-time web assistant operating in {current_year}.\n"
            f"USER QUERY: {request.message}\n\n"
            f"LIVE SEARCH RESULTS:\n{search_results}\n\n"
            "STRICT GUIDELINES:\n"
            "1. Answer using ONLY the live search results provided above.\n"
            "2. If the user asks for weather, temperature, or news, look ONLY at the weather and news facts in the search data.\n"
            "3. Do NOT mistake words like 'current' for financial platforms or brand names unless explicitly asked.\n"
            "4. NEVER mention an AI knowledge cutoff date (such as 2023).\n"
            "5. Deliver a direct, friendly, and helpful summary based on the web results."
        )

        response = llm.invoke(prompt)
        return {"reply": response.content}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

# Serve UI static files directly at root URL
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
