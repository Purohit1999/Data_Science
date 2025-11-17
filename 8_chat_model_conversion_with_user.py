# 8_chat_model_conversion_with_user.py
# -----------------------------------------------------------
# Chat loop with memory — switch between FakeLLM and OpenAI model
# -----------------------------------------------------------

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ======= FLAG SECTION =======
# Set USE_FAKE = True if you’re out of OpenAI quota or want to test offline.
USE_FAKE = True
# ============================

if USE_FAKE:
    # ---- Use FAKE model (no API key needed) ----
    from langchain_community.llms.fake import FakeListLLM
    print("⚙️ Running in FAKE MODE (no API key needed)")
    model = FakeListLLM(
        responses=[
            "(test) Hi! I'm your fake AI assistant.",
            "(test) Interesting question. Let’s pretend to analyze it!",
            "(test) Sure! That sounds great.",
        ] * 100
    )

else:
    # ---- Use REAL OpenAI model ----
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    load_dotenv()
    print("🤖 Running in REAL MODE (OpenAI API)")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Store chat messages (system + user + AI)
chat_history = []
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

print("\nType 'exit' to quit.\n")

# --------------- CHAT LOOP ---------------
while True:
    query = input("You: ").strip()
    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))

    try:
        # Generate AI response (works for both fake & real)
        result = model.invoke(chat_history)
        response = result.content if hasattr(result, "content") else result
    except Exception as e:
        print(f"⚠️ Error: {e}")
        break

    chat_history.append(AIMessage(content=response))
    print(f"AI: {response}\n")

# --------------- HISTORY SUMMARY ---------------
print("----- Chat History -----")
for msg in chat_history:
    role = msg.__class__.__name__.replace("Message", "")
    print(f"[{role}] {msg.content[:80]}")
