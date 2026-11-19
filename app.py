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

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def serve_frontend():
    index_path = "static/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="static/index.html not found")


@app.get("/manifest.json")
async def serve_manifest():
    manifest_path = "static/manifest.json"
    if os.path.exists(manifest_path):
        return FileResponse(manifest_path, media_type="application/manifest+json")
    raise HTTPException(status_code=404, detail="static/manifest.json not found")


@app.get("/sw.js")
async def serve_sw():
    sw_path = "static/sw.js"
    if os.path.exists(sw_path):
        return FileResponse(sw_path, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="static/sw.js not found")


search_tool = DuckDuckGoSearchRun()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise RuntimeError("GROQ_API_KEY environment variable is missing.")

# Replace with an active chat model ID from your Groq Console if needed
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    api_key=groq_key,
)


class MessageRequest(BaseModel):
    message: str


GREETINGS = {
    "hi", "hii", "hiii", "hello", "hey", "yo", "sup",
    "good morning", "good evening", "good night"
}


@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        msg = request.message.strip()
        if not msg:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        current_year = datetime.now().year
        is_greeting = msg.lower().rstrip("!.?") in GREETINGS

        if is_greeting:
            prompt = (
                f"Current Year: {current_year}\n"
                f"User says: {msg}\n"
                "Reply warmly and briefly as PRIME AI."
            )
            response = llm.invoke(prompt)
            return {"reply": response.content}

        search_query = f"{msg} {current_year}"
        try:
            search_results = search_tool.run(search_query)
        except Exception as search_err:
            search_results = f"Search unavailable ({str(search_err)})."

        if not search_results or not str(search_results).strip():
            search_results = "No web snippets returned."

        prompt = (
            f"Current Year: {current_year}\n"
            f"User Query: {msg}\n\n"
            f"Web Context:\n{search_results}\n\n"
            "INSTRUCTIONS:\n"
            "- Use the web context when relevant for current/recent facts.\n"
            "- Never mention a 2023 knowledge cutoff.\n"
            "- Be clear and helpful."
        )

        response = llm.invoke(prompt)
        return {"reply": response.content}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")
