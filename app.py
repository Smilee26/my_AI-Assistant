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

# 1. Enable CORS for GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Fetch Groq API Key
groq_key = os.getenv("GROQ_API_KEY")

# 3. Initialize Groq Chat Model using official Groq Model ID
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Or "llama-3.1-8b-instant" / "qwen/qwen3.6-27b"
    temperature=0.2,
    api_key=groq_key
)

# 4. Bind Search Tool to the Model
search_tool = DuckDuckGoSearchRun()
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
                content=f"You are PRIME, an intelligent AI assistant. Today's date is {today_str}. "
                        f"Always use the web search tool to retrieve live, current, and real-time information."
            ),
            HumanMessage(content=request.message)
        ]

        # First LLM execution
        ai_msg = llm_with_tools.invoke(messages)

        # Handle tool execution loop for live DuckDuckGo Search
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            messages.append(ai_msg)
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    
                    # Safely handle string vs dict queries
                    if isinstance(tool_args, dict):
                        query = tool_args.get("query", str(tool_args))
                    else:
                        query = str(tool_args)
                    
                    # Execute Search
                    tool_output = selected_tool.invoke(query)
                    
                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
            
            # Send search results back for final answer
            final_res = llm_with_tools.invoke(messages)
            return {"reply": final_res.content}

        return {"reply": ai_msg.content}

    except Exception as e:
        print(f"Backend Execution Error: {str(e)}")
        # Direct fallback to simple response without tools if tool invocation fails
        try:
            fallback_res = llm.invoke(request.message)
            return {"reply": fallback_res.content}
        except Exception as fallback_err:
            raise HTTPException(status_code=500, detail=str(fallback_err))
