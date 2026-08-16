import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain Imports for IBM watsonx & Tools
from langchain_ibm import ChatWatsonx
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

# 2. Fetch IBM Watsonx API Keys & Configuration
api_key = os.getenv("IBM_CLOUD_API_KEY", "YOUR_IBM_API_KEY")
project_id = os.getenv("IBM_PROJECT_ID", "YOUR_WATSONX_PROJECT_ID")
url = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

watsonx_params = {
    "max_new_tokens": 800,
    "temperature": 0.3
}

# 3. Initialize IBM Granite 3.0 Model via LangChain
llm = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url=url,
    apikey=api_key,
    project_id=project_id,
    params=watsonx_params
)

# 4. Bind Search Tool to IBM Granite
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        # Dynamically inject today's exact date to solve the knowledge cutoff issue
        today_str = datetime.now().strftime("%B %d, %Y")
        
        messages = [
            SystemMessage(
                content=f"You are PRIME, a helpful AI assistant. Today's date is {today_str}. "
                        "Always use the web search tool to retrieve real-time data, current events, dates, "
                        "or information beyond your training cutoff."
            ),
            HumanMessage(content=request.message)
        ]

        # First LLM invocation
        ai_msg = llm_with_tools.invoke(messages)

        # Handle tool calling loop for live DuckDuckGo Search
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
            
            # Feed search results back to IBM Granite for the final answer
            final_res = llm_with_tools.invoke(messages)
            return {"reply": final_res.content}

        return {"reply": ai_msg.content}

    except Exception as e:
        # Graceful fallback to raw Granite invocation if tool-calling fails
        try:
            fallback_res = llm.invoke(request.message)
            return {"reply": fallback_res.content}
        except Exception:
            raise HTTPException(status_code=500, detail=str(e))
