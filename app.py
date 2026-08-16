import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from googlesearch import search

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@tool
def google_search(query: str) -> str:
    """Searches Google for current news, real-time data, and up-to-date information."""
    try:
        results = list(search(query, num_results=3, advanced=True))
        if not results:
            return "No Google search results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r.title}\nSnippet: {r.description}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=groq_key
)

tools = [google_search]
tools_by_name = {tool.name: tool for tool in tools}

# FIX: Pass the explicit tool name "google_search" or "any" instead of "required"
llm_forced_tool = llm.bind_tools(tools, tool_choice="google_search")

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        current_year = datetime.now().year
        today_str = datetime.now().strftime("%B %d, %Y")

        messages = [
            SystemMessage(
                content=(
                    f"You are PRIME AI, operating in {current_year}. Today is {today_str}.\n"
                    "Use the search results to answer the query accurately with current information."
                )
            ),
            HumanMessage(content=request.message)
        ]

        # 1. Mandatory Tool Call Step
        ai_msg = llm_forced_tool.invoke(messages)

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            messages.append(ai_msg)
            
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    query_str = tool_args.get("query", request.message) if isinstance(tool_args, dict) else str(tool_args)
                    
                    search_results = selected_tool.invoke({"query": query_str})
                    
                    messages.append(
                        ToolMessage(content=str(search_results), tool_call_id=tool_call["id"])
                    )
            
            # 2. Final Synthesis Step (Standard LLM without forced tool choice)
            final_res = llm.invoke(messages)
            
            reply_text = final_res.content if final_res.content else "I have retrieved the latest results."
            return {"reply": reply_text}

        return {"reply": "Unable to execute search."}

    except Exception as e:
        print(f"Error: {str(e)}")
        # Fallback to standard generation if search execution fails
        fallback = llm.invoke(request.message)
        return {"reply": fallback.content}
