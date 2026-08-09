import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, api_key=groq_key)

search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        messages = [
            SystemMessage(
                content="You are PRIME, a helpful AI assistant. Use the web search tool if you need real-time or updated information."
            ),
            HumanMessage(content=request.message)
        ]

        # First LLM invocation
        ai_msg = llm_with_tools.invoke(messages)

        # Handle tool calling loop
        if ai_msg.tool_calls:
            messages.append(ai_msg)
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    
                    # Extract string query safely if passed inside a dict
                    query = tool_args.get("query", tool_args) if isinstance(tool_args, dict) else tool_args
                    
                    tool_output = selected_tool.invoke(query)
                    
                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
            
            # Send tool outputs back to LLM for final answer
            final_res = llm_with_tools.invoke(messages)
            return {"reply": final_res.content}

        return {"reply": ai_msg.content}

    except Exception as e:
        # Graceful fallback if tool calling or Groq execution fails
        try:
            fallback_res = llm.invoke(request.message)
            return {"reply": fallback_res.content}
        except Exception:
            raise HTTPException(status_code=500, detail=str(e))