# 12_chains_sequential.py
# -----------------------
# Facts about an animal -> translate to French -> append word count.

import os
from dotenv import load_dotenv

# Version-friendly imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda
except ImportError:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableLambda

from langchain_openai import ChatOpenAI


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment or .env")

    # Use a small, fast model by default; change if you like
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"), temperature=0)

    # --- Prompt 1: make facts about an animal ---
    animal_facts_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You like telling concise facts about {animal}."),
            ("human", "Tell me {count} facts."),
        ]
    )

    # --- Prompt 2: translate given text to a target language ---
    translation_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator and convert the provided text into {language}."),
            ("human", "Translate the following text to {language}: {text}"),
        ]
    )

    # Extra processing steps
    prepare_for_translation = RunnableLambda(
        lambda output: {"text": output, "language": "French"}
    )

    count_words = RunnableLambda(
        lambda text: f"Word count: {len(text.split())}\n\n{text}"
    )

    # Build the sequential chain
    chain = (
        animal_facts_template
        | model
        | StrOutputParser()          # String from model #1
        | prepare_for_translation    # Map to {text, language}
        | translation_template
        | model
        | StrOutputParser()          # String from model #2 (French)
        | count_words
    )

    # Run it
    result = chain.invoke({"animal": "cat", "count": 3})
    print(result)


if __name__ == "__main__":
    main()
