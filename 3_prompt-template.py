from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

model=ChatOpenAI(model="gpt-5-nano", temperature=0.0)

# Prompt template is a way to create dynamic prompts by filling in variables. 
# And it is useful when you want to create a prompt that can be reused with different inputs.

# Define the system prompt template
sys_prompt_template = PromptTemplate.from_template("""Answer the following question using the given context.
    If the context does not provide enough information, say 'I don't know'.
    Context: {context}
    Question: {question}"""
)

# generate the system prompt
sys_prompt = sys_prompt_template.invoke({
    "context": "The capital of India is New Delhi.",
    "question": "What is the capital of France?"
})

# Create the system message
response = model.invoke(sys_prompt)
print(response.content)