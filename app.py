import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize both API components
groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Set up Tavily real-time search tool
search_tool = TavilySearchResults(
    max_results=3,
    tavily_api_key=tavily_key
)

# Set up Groq LLM
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
        
        # Execute Tavily search
        try:
            raw_results = search_tool.invoke({"query": request.message})
            search_context = "\n".join([r.get("content", "") for r in raw_results])
        except Exception as search_err:
            search_context = f"Search failed: {str(search_err)}"

        if not search_context.strip():
            search_context = "No specific live web data retrieved."

        # Prompt instruction preventing cutoff responses
        prompt = (
            f"SYSTEM: You are PRIME AI operating on {today_date}.\n"
            f"USER QUERY: {request.message}\n\n"
            f"LIVE SEARCH DATA:\n{search_context}\n\n"
            "RULES:\n"
            "1. Base your response on the live search data.\n"
            "2. NEVER mention a 2023 knowledge cutoff date.\n"
            "3. Provide a clear, real-time answer."
        )

        response = llm.invoke(prompt)
        return {"reply": response.content}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")
