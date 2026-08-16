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

# 3. Initialize Groq Chat Model
llm = ChatGroq(
    model="qwen3.6-27b",
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
        # Dynamically calculate current 2026 date
        today_date = datetime.now()
        today_str = today_date.strftime("%B %d, %Y")
        current_year = today_date.year

        # System message forcing 2026 baseline and overriding static cutoff answers
        messages = [
            SystemMessage(
                content=(
                    f"You are PRIME, an intelligent AI assistant operating in {current_year}. "
                    f"Today's date is {today_str}. "
                    f"Never say your knowledge ends in December 2023. You have real-time web access for {current_year}. "
                    f"If the user asks about your knowledge cutoff, state that you operate with live 2026 data via web search. "
                    f"For any queries regarding current news, events, dates, or real-time topics, ALWAYS execute "
                    f"a web search to get accurate, up-to-date results before answering."
                )
            ),
            HumanMessage(content=request.message)
        ]

        # First LLM execution
        ai_msg = llm_with_tools.invoke(messages)

        # Handle tool execution loop for live DuckDuckGo Search
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
            
            # Send web search results back to the LLM for the final answer
            final_res = llm_with_tools.invoke(messages)
            return {"reply": final_res.content}

        return {"reply": ai_msg.content}

    except Exception as e:
        # Fallback to direct invocation if tool routing encounters an error
        try:
            fallback_res = llm.invoke(request.message)
            return {"reply": fallback_res.content}
        except Exception:
            raise HTTPException(status_code=500, detail=str(e))
