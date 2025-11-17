# 10_chains_basics.py
# ---------------------------------------------------------------
# Minimal LCEL demo:
#   ChatPromptTemplate  ->  ChatOpenAI  ->  StrOutputParser
# Works with langchain>=0.2, langchain-openai, and python-dotenv.
# Adds:
#   • .env loading (OPENAI_API_KEY)
#   • model/temperature overrides via env
#   • fake mode when no key is available
# ---------------------------------------------------------------

import os
from dotenv import load_dotenv

# LangChain v0.2+ imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ---------- Load environment ----------
load_dotenv()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

# Optional overrides
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

FAKE_MODE = not bool(OPENAI_API_KEY)  # run without calling OpenAI if no key
if FAKE_MODE:
    print("⚙️ Running in FAKE MODE (no API key detected).")

# ---------- Build model (or fake) ----------
def build_model():
    if FAKE_MODE:
        # Return a tiny shim that mimics .invoke() and .generate()
        class _Fake:
            def invoke(self, messages, **kwargs):
                # messages is a list of dicts from the prompt template;
                # we just produce a playful deterministic response.
                return type("Resp", (), {"content": "🐘 Elephants are the largest land animals."})
        return _Fake()

    # Real OpenAI-backed model
    return ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

llm = build_model()

# ---------- Prompt: system + human messages ----------
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise facts expert about {animal}."),
        ("human", "Give me {fact_count} short fact(s)."),
    ]
)

# ---------- Parser: convert model output to plain string ----------
output_parser = StrOutputParser()

# ---------- Chain: prompt -> model -> parser ----------
chain = prompt_template | llm | output_parser

# ---------- Run the chain ----------
inputs = {"animal": "elephant", "fact_count": 1}
result = chain.invoke(inputs)

print("---- Result ----")
print(result)

# Tip:
#   Put your key in a .env file at project root like:
#       OPENAI_API_KEY=sk-...
#   (Optionally)
#       OPENAI_MODEL=gpt-4o-mini
#       OPENAI_TEMPERATURE=0.3
