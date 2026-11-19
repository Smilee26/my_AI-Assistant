import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# Set Web Page Title
st.set_page_config(page_title="My AI Agent", page_icon="🤖")
st.title("🤖 My Custom AI Agent")

# Retrieve API Key from Environment Variables (Set on hosting platform)
groq_api_key = os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY is missing! Please set it in your environment/secrets.")
    st.stop()

# Initialize LLM & Tools
@st.cache_resource
def load_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=groq_api_key
    )
    search_tool = DuckDuckGoSearchRun()
    tools = [search_tool]

    template = """You are a helpful AI Assistant. You have access to the following tools:

{tools}

To use a tool, use the format:
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response, or if you don't need a tool:
Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Previous Conversation History:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

agent_executor = load_agent()

# Session Memory Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask your agent something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Format chat history text
    chat_history_str = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages[:-1]
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history_str
            })
            answer = response["output"]
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
