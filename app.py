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
    """Searches Google for real-time information, news, current events, and up-to-date data."""
    try:
        results = list(search(query, num_results=4, advanced=True))
        if not results:
            return "No search results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r.title}\nSnippet: {r.description}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search execution failed: {str(e)}"

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    api_key=groq_key
)

tools = [google_search]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        current_year = datetime.now().year
        today_str = datetime.now().strftime("%B %d, %Y")

        # System prompt overriding default internal cutoff assumptions
        messages = [
            SystemMessage(
                content=(
                    f"You are PRIME AI, an up-to-date AI assistant operating in the year {current_year}. Today is {today_str}.\n\n"
                    "RULES FOR UPDATED INFORMATION:\n"
                    "1. Never mention a static '2023 knowledge cutoff'. You have live access to current information.\n"
                    "2. ALWAYS call the `google_search` tool if the user asks about:\n"
                    "   - Current dates, knowledge cutoffs, or operational capabilities.\n"
                    "   - Recent news, sports, tech releases, or events beyond 2023.\n"
                    "3. Once you receive search results, synthesize them into a clear, direct, and complete textual answer."
                )
            ),
            HumanMessage(content=request.message)
        ]

        # 1. First execution: Let LLM select/trigger tools
        ai_msg = llm_with_tools.invoke(messages)

        # 2. If the LLM generated tool calls
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
            
            # 3. Second execution: Run standard LLM to synthesize final response
            final_res = llm.invoke(messages)
            reply_text = final_res.content if final_res.content else "I have retrieved the latest updates for 2026."
            return {"reply": reply_text}

        # Handle direct output when no tools are invoked
        reply_text = ai_msg.content if ai_msg.content else "How can I assist you with current information today?"
        return {"reply": reply_text}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")
