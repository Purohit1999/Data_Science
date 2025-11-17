# 13_chains_parallel.py
# ---------------------
# Movie summary -> (plot analysis || character analysis) in parallel -> combined report

import os
from dotenv import load_dotenv

# Version-friendly imports (LangChain 0.1.x–0.3.x)
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda, RunnableParallel
except ImportError:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableLambda, RunnableParallel

from langchain_openai import ChatOpenAI


def main():
    # --- Env ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment or .env")

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"), temperature=0)

    # --- Step 1: Summarize the movie ---
    summary_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Provide a brief summary of the movie {movie_name}."),
        ]
    )

    summarize_chain = summary_template | model | StrOutputParser()

    # --- Step 2a: Plot analysis branch ---
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),
        ]
    )
    plot_branch_chain = plot_template | model | StrOutputParser()

    # --- Step 2b: Character analysis branch ---
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),
        ]
    )
    character_branch_chain = character_template | model | StrOutputParser()

    # Map the summary text to inputs for both branches
    to_branches_input = RunnableLambda(lambda summary_text: {"plot": summary_text, "characters": summary_text})

    # Run branches in parallel
    parallel = RunnableParallel(
        {
            "plot_analysis": plot_branch_chain,
            "character_analysis": character_branch_chain,
        }
    )

    # --- Step 3: Combine results ---
    def combine(report):
        plot_part = report["plot_analysis"]
        char_part = report["character_analysis"]
        return (
            "==================== FINAL REVIEW ====================\n\n"
            "🎬 Plot Analysis:\n"
            f"{plot_part}\n\n"
            "🧑‍🤝‍🧑 Character Analysis:\n"
            f"{char_part}\n"
            "=======================================================\n"
        )

    combine_results = RunnableLambda(combine)

    # --- Full chain: summary -> parallel analyses -> combine ---
    chain = summarize_chain | to_branches_input | parallel | combine_results

    # Run it
    result = chain.invoke({"movie_name": "Inception"})
    print(result)


if __name__ == "__main__":
    main()
