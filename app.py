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
    """Searches Google for real-time information and returns top results."""
    try:
        results = list(search(query, num_results=5, advanced=True))
        if not results:
            return "No search results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r.title}\nSnippet: {r.description}\nURL: {r.url}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search failed: {str(e)}"

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=groq_key
)

tools = [google_search]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        today_date = datetime.now()
        today_str = today_date.strftime("%B %d, %Y")
        current_year = today_date.year

        messages = [
            SystemMessage(
                content=(
                    f"You are PRIME AI, an assistant operating in {current_year}. Today is {today_str}.\n"
                    f"CRITICAL DIRECTIVE:\n"
                    f"- If the user asks about knowledge cutoffs, dates, current news, sports, or real-time info, "
                    f"you MUST call the google_search tool FIRST.\n"
                    f"- Do NOT state your static knowledge cutoff is 2023. Answer based on live search results for {current_year}."
                )
            ),
            HumanMessage(content=request.message)
        ]

        # First LLM Call
        ai_msg = llm_with_tools.invoke(messages)

        # Handle tool calling loop
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            messages.append(ai_msg)
            
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    query = tool_args.get("query", request.message) if isinstance(tool_args, dict) else str(tool_args)
                    
                    try:
                        tool_output = selected_tool.invoke({"query": query})
                    except Exception as err:
                        tool_output = f"Search fallback: {str(err)}"
                    
                    messages.append(
                        ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                    )
            
            # Synthesize live 2026 data into final answer
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
