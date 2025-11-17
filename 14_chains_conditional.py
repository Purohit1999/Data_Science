# 14_chains_conditional.py
# ------------------------
# Classify a piece of feedback and route to the right response chain.

import os
from dotenv import load_dotenv

# Version-friendly imports (works for 0.1.x–0.3.x)
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableBranch
except ImportError:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableBranch

from langchain_openai import ChatOpenAI


def main():
    # --- Env ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in your environment or .env")

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5-nano"), temperature=0)

    # ------------------ Prompts ------------------

    # Response generators for each sentiment
    positive_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Generate a warm thank-you note for this positive feedback: {feedback}"),
        ]
    )

    negative_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Generate an empathetic response addressing this negative feedback: {feedback}"),
        ]
    )

    neutral_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Ask for more details about this neutral/ambiguous feedback: {feedback}"),
        ]
    )

    escalate_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "Draft a brief escalation message to a human agent about this feedback: {feedback}"),
        ]
    )

    # Classification prompt (decide positive/negative/neutral/escalate)
    classification_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Classify the sentiment of this feedback as exactly one of: "
                "positive, negative, neutral, or escalate. Return only the label.\n\nFeedback: {feedback}"
            ),
        ]
    )

    # ------------------ Chains ------------------

    # 1) Classify
    classify_chain = classification_template | model | StrOutputParser()

    # 2) Conditional router
    # Each branch receives the ORIGINAL input (e.g., {'feedback': '...'})
    # RunnableBranch picks the first condition that evaluates to True.
    def has_label(label):
        return lambda x: label in x["label"].lower()

    router = RunnableBranch(
        # Positive
        (
            has_label("positive"),
            positive_feedback_template | model | StrOutputParser(),
        ),
        # Negative
        (
            has_label("negative"),
            negative_feedback_template | model | StrOutputParser(),
        ),
        # Neutral
        (
            has_label("neutral"),
            neutral_feedback_template | model | StrOutputParser(),
        ),
        # Default / escalate
        escalate_feedback_template | model | StrOutputParser(),
    )

    # 3) Wire it up: classify -> attach label -> route
    # We first compute the label, then pass both label and original feedback to the router.
    def to_router_input(x):
        # x is {'feedback': '...'} at this stage
        label = classify_chain.invoke(x)
        return {"label": label, **x}

    # Run
    feedback = "The product is terrible. It broke after one use and the quality is poor."
    result = router.invoke(to_router_input({"feedback": feedback}))
    print("Label & response\n-----------------")
    print(result)


if __name__ == "__main__":
    main()
