import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        
        # 1. Force search execution
        search_query = f"{request.message} {current_year}"
        try:
            search_results = search_tool.run(search_query)
        except Exception as search_err:
            search_results = f"No live search results available. Details: {str(search_err)}"

        if not search_results or len(search_results.strip()) == 0:
            search_results = "Search executed but returned no text."

        # 2. Hardcode prompt constraints to prevent 2023 fallback
        prompt = (
            f"Current Year: {current_year}\n"
            f"User Query: {request.message}\n\n"
            f"Web Context:\n{search_results}\n\n"
            "STRICT RULES:\n"
            "- Answer the user using ONLY the web context provided above.\n"
            "- NEVER say 'As of my knowledge cutoff in 2023'.\n"
            "- If context is limited, summarize what is available for 2026 without mentioning knowledge cutoffs."
        )

        response = llm.invoke(prompt)
        return {"reply": response.content}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")
