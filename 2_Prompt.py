from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage


load_dotenv()

# System message to set the behavior of the assistant
# Human message to ask a question
# AI message to respond to the question

model=ChatOpenAI(model="gpt-5-nano", temperature=0.0)
sys_prompt=SystemMessage(
    content="""You are a helpful assistant that answers questions about current events and general knowledge for the children in the age group 8 to 12 years.
    Ensure that your responses are simple, clear, and age-appropriate.
    If you don't know the answer, say 'I don't know' instead of making up an answer.
    Always provide accurate and factual information."""
)
human_prompt=HumanMessage(
    content="What is the capital of France?")

response = model.invoke([sys_prompt, human_prompt])
print(response.content)