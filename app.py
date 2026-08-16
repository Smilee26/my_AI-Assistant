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
    """Searches Google for current events, news, and real-time information."""
    try:
        results = list(search(query, num_results=4, advanced=True))
        if not results:
            return "No Google search results found."
        
        output = []
        for r in results:
            output.append(f"Title: {r.title}\nSnippet: {r.description}\nURL: {r.url}")
        return "\n\n".join(output)
    except Exception as e:
        return f"Search temporary error: {str(e)}"

groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    api_key=groq_key
)

tools = [google_search]
tools_by_name = {tool.name: tool for tool in tools}

# Bind tool to LLM
llm_with_tools = llm.bind_tools(tools)

class MessageRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat_endpoint(request: MessageRequest):
    try:
        today_str = datetime.now().strftime("%B %d, %Y")

        # 1. Provide system context overriding default cutoff assumptions
        messages = [
            SystemMessage(
                content=(
                    f"You are PRIME AI, an up-to-date AI assistant. Today's date is {today_str}.\n"
                    "INSTRUCTIONS:\n"
                    "1. For questions about current events, news, recent updates, or your cutoff date, "
                    "you MUST call the 'google_search' tool.\n"
                    "2. Always synthesize the search results into a detailed, complete final textual answer."
                )
            ),
            HumanMessage(content=request.message)
        ]

        # 2. First turn: LLM decides whether to run a tool
        ai_msg = llm_with_tools.invoke(messages)

        # 3. Check if tool calls exist
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            # Append the assistant's tool-call intention message to conversation memory
            messages.append(ai_msg)
            
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})
                
                if tool_name in tools_by_name:
                    selected_tool = tools_by_name[tool_name]
                    query_str = tool_args.get("query", request.message) if isinstance(tool_args, dict) else str(tool_args)
                    
                    # Execute Search
                    search_results = selected_tool.invoke({"query": query_str})
                    
                    # Append Tool Result back to memory
                    messages.append(
                        ToolMessage(content=str(search_results), tool_call_id=tool_call["id"])
                    )
            
            # 4. Second turn: Run standard LLM (without tool binding) to synthesize the answer
            final_res = llm.invoke(messages)
            
            # Prevent empty response bugs
            reply_content = final_res.content if final_res.content else "I couldn't generate a text response based on the search data."
            return {"reply": reply_content}

        # Handle direct answers where search wasn't triggered
        reply_content = ai_msg.content if ai_msg.content else "How can I assist you today?"
        return {"reply": reply_content}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
