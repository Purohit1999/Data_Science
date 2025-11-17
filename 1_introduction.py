from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI


load_dotenv()

model=ChatOpenAI(model="gpt-5-nano", temperature=0.0, max_tokens=1000)

response = model.invoke("What is the capital of India?")
print(response)
print(response.content)