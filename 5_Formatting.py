# ---------------------------------- Imports ----------------------------------
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# -------------------------------- Load .env ---------------------------------
load_dotenv()  # Loads environment variables from .env file

# ---------------------- Define Pydantic model for structured output ----------------------
class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director of the movie")
    year: int = Field(description="The release year of the movie")
    description: str = Field(description="Brief description or summary of the movie")

# ---------------------- Initialize the chat model ----------------------
# Using gpt-5-nano as per your code (or you can replace with 'gpt-4' or 'gpt-3.5-turbo')
model = ChatOpenAI(model="gpt-5-nano", temperature=1)

# ---------------------- Create output parser ----------------------
parser = PydanticOutputParser(pydantic_object=Movie)

# ---------------------- Define prompt template ----------------------
template = PromptTemplate(
    input_variables=["context", "question"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
    template="""
You are a helpful assistant that answers questions about movies.

Use the context if available, but if the context does not fully answer, generate additional details based on general knowledge.

Context: {context}
Question: {question}

Answer in JSON format:
{format_instructions}
"""
)

# ---------------------- Build the chain using RunnableSequence style ----------------------
chain = template | model | parser

# ---------------------- Example input ----------------------
user_input = {
    "context": "The movie 'Inception' is directed by Christopher Nolan and was released in 2010.",
    "question": "Give me the full details of the movie including a short summary."
}

# ---------------------- Generate response ----------------------
movie_struct = chain.invoke(user_input)

# ---------------------- Print results ----------------------
print("\nStructured JSON Output:\n", movie_struct.model_dump_json(indent=2))
