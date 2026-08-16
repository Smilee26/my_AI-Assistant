import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
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

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=groq_key
)

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
                        f"Operate fully within the context of {datetime.now().year}. "
                        f"If asked about current events, dates, or live topics, use the web search tool."
            ),
            HumanMessage(content=request.message)
        ]

        # 1. First LLM Execution
        ai_msg = llm_with_tools.invoke(messages)

        # 2. Check for Tool Calls
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            messages.append(ai_msg)
            
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    
                    # Extract query safely
                    if isinstance(tool_args, dict):
                        query = tool_args.get("query", str(tool_args))
                    else:
                        query = str(tool_args)
                    
                    # Safe tool execution (prevents 500 errors on search failures)
                    try:
                        tool_output = selected_tool.invoke(query)
                    except Exception as tool_err:
                        tool_output = f"Search currently unavailable ({str(tool_err)}). Answer using system instructions."
                    
                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
            
            # 3. Final response generation
            final_res = llm_with_tools.invoke(messages)
            return {"reply": final_res.content}

        return {"reply": ai_msg.content}

    except Exception as e:
        print(f"Backend Error: {str(e)}")
        # Safe fallback: invoke model directly without tool routing
        try:
            fallback_res = llm.invoke(request.message)
            return {"reply": fallback_res.content}
        except Exception as final_err:
            raise HTTPException(status_code=500, detail=str(final_err))
