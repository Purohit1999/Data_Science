# 11_chains_inner_workings.py
# ---------------------------
# A minimal LangChain "chain internals" demo using RunnableLambda + RunnableSequence.

import os
from dotenv import load_dotenv

# --- Version-friendly imports (works for 0.1.x–0.3.x) ---
try:
    # Newer LangChain (0.2+)
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableSequence
except ImportError:
    # Older LangChain (0.1.x)
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnableLambda, RunnableSequence

from langchain_openai import ChatOpenAI


def main():
    # STEP 1: Load environment (.env should define OPENAI_API_KEY)
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Please set OPENAI_API_KEY in a .env file or environment variable.")

    # STEP 2: Initialize the model
    # Use a small, fast model; change if desired.
    model = ChatOpenAI(model="gpt-5-nano", temperature=0)

    # STEP 3: Define the prompt template (placeholders: {animal}, {count})
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You love facts and you tell facts about {animal}."),
            ("human", "Tell me {count} facts."),
        ]
    )

    # STEP 4: Define individual runnables (chain steps)
    # 4.1 Format the prompt with provided inputs
    format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

    # 4.2 Invoke the model with the formatted messages
    invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))

    # 4.3 Extract only the text content from the model response
    parse_output = RunnableLambda(lambda x: getattr(x, "content", str(x)))

    # STEP 5: Create a runnable sequence (pipeline)
    chain = RunnableSequence(
        first=format_prompt,       # Step 1: format prompt
        middle=[invoke_model],     # Step 2: call the model
        last=parse_output          # Step 3: parse to plain text
    )

    # STEP 6: Run the chain (provide values for {animal} and {count})
    response = chain.invoke({"animal": "cat", "count": 2})

    # STEP 7: Print final output
    print(response)


if __name__ == "__main__":
    main()
