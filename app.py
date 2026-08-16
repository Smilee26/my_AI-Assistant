import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports for Groq & Tools
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

load_dotenv()

app = FastAPI()

# 1. CORS Configuration for GitHub Pages Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Fetch Groq API Key
groq_key = os.getenv("GROQ_API_KEY")

# 3. Initialize Groq Chat Model (Using active replacement model)
llm = ChatGroq(
    model="qwen3.6-27b",  # Or "gpt-oss-120b"
    temperature=0.2,
    api_key=groq_key
)

# 4. Bind DuckDuckGo Web Search Tool
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        # Dynamically fetch current date (e.g., August 16, 2026)
        today_date = datetime.now()
        today_str = today_date.strftime("%B %d, %Y")
        current_year = today_date.year

        # System message forcing 2026 temporal awareness and search execution
        messages = [
            SystemMessage(
                content=(
                    f"You are PRIME, an intelligent AI assistant. Today's date is {today_str} (Year {current_year}). "
                    f"Your static training data ends in the past. Whenever a user asks about current events, "
                    f"dates, live information, or anything taking place in {current_year}, you MUST invoke "
                    f"the web search tool to retrieve accurate, up-to-date data before responding. "
                    f"Never say your knowledge ends in past years without searching first."
                )
            ),
            HumanMessage(content=request.message)
        ]

        # First LLM execution
        ai_msg = llm_with_tools.invoke(messages)

        # Handle tool calling loop for DuckDuckGo
        if ai_msg.tool_calls:
            messages.append(ai_msg)
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    query = tool_args.get("query", tool_args) if isinstance(tool_args, dict) else tool_args
                    
                    # Execute Search
                    tool_output = selected_tool.invoke(query)
                    
                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
            
            # Send search results back to Groq model for final 2026 answer
            final_res = llm_with_tools.invoke(messages)
            return {"reply": final_res.content}

        return {"reply": ai_msg.content}

    except Exception as e:
        # Graceful fallback to direct LLM execution if tool calling fails
        try:
            fallback_res = llm.invoke(request.message)
            return {"reply": fallback_res.content}
        except Exception:
            raise HTTPException(status_code=500, detail=str(e))
