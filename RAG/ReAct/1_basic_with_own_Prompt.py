"""
ReAct Agent (Reason + Act) with a custom tool in LangChain.

What this script does
---------------------
1) Loads your OpenAI API key from .env
2) Creates a ChatOpenAI LLM
3) Defines a @tool: get_system_time(format=...) that ALWAYS returns IST time
4) Builds a custom ReAct-style prompt (with {tools}, {tool_names}, {agent_scratchpad}, {input})
5) Creates a ReAct agent + AgentExecutor
6) Runs a sample query

Requirements
-----------
pip install langchain langchain-openai python-dotenv
# (Python 3.9+ recommended; uses zoneinfo from stdlib)

.env file (same folder)
-----------------------
OPENAI_API_KEY=sk-...
"""

from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool  # tool decorator (works in recent LangChain)

# 1) Load environment variables (OPENAI_API_KEY)
load_dotenv()

# 2) Initialize the LLM
#    You can change to "gpt-4o" / "gpt-4.1" / "gpt-4o-mini" if you like.
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3) Define Tools
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current time in **India Standard Time (IST, UTC+05:30)** using the given strftime `format`.
    Example formats:
      - "%Y-%m-%d %H:%M:%S"
      - "%I:%M %p"
      - "%a, %d %b %Y %H:%M"
    """
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    return now_ist.strftime(format)

# You can register more tools here if needed, e.g., calculators, web search, DB queries, etc.
tools = [get_system_time]

# 4) Build a custom ReAct prompt
custom_prompt = """
You are a helpful AI agent.

You have access to the following tools:
{tools}

You are currently running in **India Standard Time (IST)**.
If the user asks for the time in another city, you must:
1) Use the 'get_system_time' tool to fetch the current IST time.
2) Convert the time to the requested city by applying the timezone difference manually,
   and clearly state both the source (IST) and target timezone in your answer.

Use the following format when reasoning:

Question: the input question
Thought: reasoning about what to do
Action: the action to take (one of [{tool_names}])
Action Input: the input to the action
Observation: the result of the action
... (repeat if needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
{agent_scratchpad}
""".strip()

prompt = PromptTemplate.from_template(custom_prompt)

# 5) Create the ReAct agent + executor
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6) Run a sample query
if __name__ == "__main__":
    # Try any of these:
    # query = "What is the current time in IST (no date)? Use HH:MM only."
    # query = "What's the current time now? Show as 12-hour clock with AM/PM."
    query = "Give the current time in Toronto only (no date)."
    result = agent_executor.invoke({"input": query})
    print("\nAgent final output:", result)
