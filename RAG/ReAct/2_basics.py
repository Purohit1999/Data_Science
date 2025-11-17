"""
2_basics.py — ReAct agent with a custom time-by-city tool.

What this script shows:
1) Load secrets from .env
2) Build an LLM (OpenAI via LangChain)
3) Create a timezone-aware tool `get_system_time_by_city`
4) Plug tool + LLM into a standard ReAct prompt from LangChain Hub
5) Run the agent on a natural-language question

Setup
-----
pip install -U langchain langchain-openai langchain-community python-dotenv pytz
# (choose a model you have access to; e.g., gpt-4o-mini)

.env (same folder)
------------------
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
"""

# -------------------------------
# Imports and environment
# -------------------------------
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

import datetime
import pytz  # IANA timezones helper

# The @tool decorator moved; support both locations for compatibility.
try:
    from langchain.tools import tool  # newer versions
except Exception:
    from langchain.agents import tool  # older versions


# -----------------------------------------
# Load environment variables from .env
# -----------------------------------------
# Typical usage: .env contains OPENAI_API_KEY=...
# Call before creating ChatOpenAI so the key is in the environment.
load_dotenv()


# -----------------------------------------
# Step 1: LLM Setup
# -----------------------------------------
# Instantiate the ChatOpenAI LLM wrapper used by LangChain.
# model="gpt-4o-mini" is fast and cost-effective; change if needed.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# temperature=0 => deterministic, helpful for demos/tests.


# --------------------------------------------------------
# Step 2: Helper — City name -> IANA timezone string map
# --------------------------------------------------------
# Keep this mapping explicit to avoid ambiguity (many cities repeat).
CITY_TO_TIMEZONE = {
    "Toronto": "America/Toronto",
    "New York": "America/New_York",
    "Delhi": "Asia/Kolkata",
    "London": "Europe/London",
    "Tokyo": "Asia/Tokyo",
    "Sydney": "Australia/Sydney",
    "Dubai": "Asia/Dubai",
    "Paris": "Europe/Paris",
}


def get_timezone_from_city(city_name: str) -> str:
    """
    Return an IANA timezone string given a city name using CITY_TO_TIMEZONE.

    If the city is not found, return "UTC" as a safe default.

    Input:
      city_name: human-readable name, e.g., "Toronto".  (Case-sensitive here.)
                 You could normalize with .title() if you want.
    Output:
      String like "America/Toronto".
    """
    return CITY_TO_TIMEZONE.get(city_name, "UTC")


# -----------------------------------------
# Step 3: Define the Tool
# -----------------------------------------
@tool
def get_system_time_by_city(city: str = "Delhi", format: str = "%H:%M:%S") -> str:
    """
    Return the current time (as a formatted string) for the given 'city'.

    Parameters:
      city   : e.g., "Toronto", "London" (must exist in CITY_TO_TIMEZONE map)
      format : strftime format, default "%H:%M:%S" -> "23:59:59"

    Returns:
      A formatted string representing the local time in that city,
      or a readable error if the timezone is unknown.

    Notes:
      - This function is registered as a LangChain Tool via @tool.
      - The agent can decide to call it when it needs time data.
      - Uses timezone-aware datetimes (pytz) to avoid host-machine bias.
    """
    tz_name = get_timezone_from_city(city)  # e.g., "America/Toronto"
    try:
        tz = pytz.timezone(tz_name)  # get tzinfo object
    except pytz.UnknownTimeZoneError:
        # Defensive: return a helpful error if mapping or tz DB is wrong
        return f"Unknown timezone for city: {city}"

    # timezone-aware 'now' in that city
    current_time = datetime.datetime.now(tz)

    # Format into a string using caller's requested format
    # Examples:
    #   "%H:%M:%S"   -> "23:59:59"
    #   "%I:%M %p"   -> "11:59 PM"
    #   "%Y-%m-%d %H:%M:%S" -> "2025-11-12 23:59:59"
    return current_time.strftime(format)


# -----------------------------------------
# Step 4: Setup agent and tools
# -----------------------------------------
# Register tools the agent can use (list of callables decorated with @tool).
tools = [get_system_time_by_city]

# Pull a standard ReAct prompt from LangChain Hub.
# "hwchase17/react" is a widely used community template that includes the
# Thought / Action / Observation scaffolding.
prompt_template = hub.pull("hwchase17/react")

# Create the ReAct agent:
# - llm:        the LLM used to "think"
# - tools:      functions the agent may call
# - prompt:     governs how the agent writes Thoughts/Actions/Observations
agent = create_react_agent(llm, tools, prompt_template)

# -----------------------------------------
# Step 5: Create executor (with parsing safety)
# -----------------------------------------
# AgentExecutor runs the reasoning loop, calls tools, and returns the final text.
# handle_parsing_errors=True makes the executor more robust if the LLM outputs
# slightly off-format actions.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,               # print Thoughts / Actions / Observations to console
    handle_parsing_errors=True,
)


# -----------------------------------------
# Step 6: Invoke agent with a natural-language query
# -----------------------------------------
if __name__ == "__main__":
    # Feel free to change this; agent will choose whether to call the tool.
    query = "Get the current time in Toronto only (no date)"
    result = agent_executor.invoke({"input": query})

    # Depending on LangChain version, result is usually a dict with key "output".
    print("\n✅ FINAL ANSWER:")
    print(result.get("output", result))

    # --- Optional: direct tool call for unit testing / quick check ---
    # print(get_system_time_by_city(city="Toronto", format="%H:%M:%S"))
