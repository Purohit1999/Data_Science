# 8_rag_chain.py
"""
Minimal RAG with FAISS + LangChain (modern imports) + OpenAI.
- Loads a PDF
- Splits text
- Builds FAISS index
- Retrieves top-k chunks
- Answers with ChatOpenAI using only retrieved context
"""

import os
from dotenv import load_dotenv

# --- LangChain modern imports ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_core.runnables import chain

# -------------------------------
# Config
# -------------------------------

PDF_PATH = os.environ.get("RAG_PDF_PATH", "./Arjun_Varma_Generative_AI_Resume.pdf")
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")   # set to any available chat model
TOP_K = int(os.environ.get("RAG_TOP_K", "2"))
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "50"))
SAVE_FAISS = os.environ.get("RAG_SAVE_FAISS", "false").lower() == "true"
FAISS_DIR = os.environ.get("RAG_FAISS_DIR", "faiss_index")

# -------------------------------
# Bootstrap
# -------------------------------

def build_vectorstore(pdf_path: str) -> FAISS:
    """Parse PDF, chunk text, and build an in-memory FAISS index."""
    # read PDF into Documents
    blob = Blob.from_path(pdf_path)
    parser = PyPDFParser()
    documents_iter = parser.lazy_parse(blob)
    docs = [d for d in documents_iter]

    # split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # embed + index
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    if SAVE_FAISS:
        vector_store.save_local(FAISS_DIR)

    return vector_store


def make_prompt() -> ChatPromptTemplate:
    """Prompt that strictly answers from context."""
    return ChatPromptTemplate.from_template(
        """Answer the question based only on the provided context.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{question}"""
    )


def main():
    load_dotenv()

    # 1) Build FAISS
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")
    vector_store = build_vectorstore(PDF_PATH)

    # 2) Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # 3) LLM + prompt
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.5)
    prompt = make_prompt()

    # 4) Simple “manual” call (for reference)
    query = "What is the experience in generative AI?"
    docs = retriever.invoke(query)

    print("Retrieved Context is as follows:")
    for i, d in enumerate(docs, start=1):
        print(f"\n--- Context {i} ---")
        print(d.page_content[:1000])  # truncate for display

    user_input = {"context": docs, "question": query}
    result = (prompt | llm).invoke(user_input)
    print("\nOutput from RAG (manual compose):")
    print(result.content)

    # 5) Runnable pipeline using @chain
    @chain
    def rag_pipeline(question: str) -> str:
        ctx_docs = retriever.invoke(question)
        final = (prompt | llm).invoke({"context": ctx_docs, "question": question})
        return final.content

    result2 = rag_pipeline.invoke(query)
    print("\nOutput from RAG (@chain pipeline):")
    print(result2)


if __name__ == "__main__":
    main()
