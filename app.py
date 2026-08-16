import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_key = os.getenv("GROQ_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Updated to an active Groq model ID
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Active replacement model on Groq
    temperature=0.2,
    api_key=groq_key
)

# Initialize Tavily Search
search_tool = TavilySearch(max_results=3)
tools = [search_tool]
tools_by_name = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        today_str = datetime.now().strftime("%B %d, %Y")

        messages = [
            SystemMessage(
                content=(
                    f"You are PRIME, an intelligent AI assistant. Today's date is {today_str}.\n"
                    "1. You have live real-time web search capabilities via Tavily.\n"
                    "2. ALWAYS invoke the search tool for questions regarding current news, events, dates, or your knowledge cutoff."
                )
            ),
            HumanMessage(content=request.message)
        ]

        ai_msg = llm_with_tools.invoke(messages)

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            messages.append(ai_msg)
            
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    
                    try:
                        tool_output = selected_tool.invoke(tool_args)
                    except Exception as err:
                        tool_output = f"Search currently unavailable: {str(err)}"
                    
                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
            
            final_res = llm_with_tools.invoke(messages)
            return {"reply": final_res.content}

        return {"reply": ai_msg.content}

    except Exception as e:
        print(f"Server Error: {str(e)}")
        try:
            fallback_res = llm.invoke(request.message)
            return {"reply": fallback_res.content}
        except Exception as final_err:
            raise HTTPException(status_code=500, detail=str(final_err))
