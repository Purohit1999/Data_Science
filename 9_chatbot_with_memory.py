# 9_chatbot_with_memory.py
# ---------------------------------------------------------------
# 🧠 Chatbot using Streamlit + LangChain + OpenAI (with memory)
# Works with LangChain v0.2+ and langchain-openai.
# Falls back to a fake reply if no OPENAI_API_KEY is set.
# ---------------------------------------------------------------

import os
import streamlit as st
from dotenv import load_dotenv

# LangChain v0.2+ imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---------- Load environment (.env must include OPENAI_API_KEY) ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ---------- Streamlit page config ----------
st.set_page_config(page_title="🧠 AI Chatbot", page_icon="💬", layout="centered")

# ---------- Sidebar controls ----------
st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox(
    "Model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
fake_mode = st.sidebar.checkbox(
    "Run without API (fake replies)",
    value=(OPENAI_API_KEY == "")
)
st.sidebar.caption("Tip: Put `OPENAI_API_KEY=sk-...` in a `.env` file at your project root.")

# ---------- Build model (or fake) ----------
def build_model():
    if fake_mode:
        return None  # we’ll generate fake replies below
    # pass the key explicitly so it works even if the env var changes at runtime
    return ChatOpenAI(api_key=OPENAI_API_KEY, model=model_name, temperature=temperature)

model = build_model()

# ---------- App title ----------
st.title("🧠 AI Chatbot")

# ---------- Initialize chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful AI assistant.")]

# ---------- Clear chat button ----------
if st.button("🧹 Clear Chat"):
    st.session_state.messages = [SystemMessage(content="You are a helpful AI assistant.")]
    st.rerun()

# ---------- Render conversation ----------
def render_history():
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(msg.content)
        # SystemMessage is not shown

render_history()

# ---------- Fake response (no API key mode) ----------
def fake_response(user_text: str) -> str:
    return (
        "🤖 *(fake reply — no API key)*\n\n"
        f"I received: **{user_text}**\n\n"
        "Add `OPENAI_API_KEY` to your `.env` to use the real model."
    )

# ---------- Handle new user input ----------
prompt = st.chat_input("Type your message…")
if prompt:
    # Add user message to history and render immediately
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    if fake_mode:
        ai_text = fake_response(prompt)
    else:
        try:
            result = model.invoke(st.session_state.messages)
            ai_text = result.content
        except Exception as e:
            ai_text = (
                f"⚠️ Error: {e}\n\n"
                "(Enable fake mode in the sidebar or check your API key/billing.)"
            )

    # Store and display AI response
    st.session_state.messages.append(AIMessage(content=ai_text))
    with st.chat_message("assistant"):
        st.markdown(ai_text)
