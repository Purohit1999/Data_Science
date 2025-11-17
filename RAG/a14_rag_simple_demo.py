# a14_rag_simple_demo.py

from dotenv import load_dotenv

from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate  # kept in case you add an LLM later


# 1. Load environment variables (useful if you add APIs later)
load_dotenv()


# 2. Postgres / pgvector connection string
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"


# 3. Embeddings model (FREE, local HuggingFace)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Load the PDF as a Blob
blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

parser = PyPDFParser()
documents_iter = parser.lazy_parse(blob)

docs = [doc for doc in documents_iter]

# 5. Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,   # small so "AI Engineer..." is its own chunk
    chunk_overlap=20,
)

chunks = splitter.split_documents(docs)

print(f"Number of chunks: {len(chunks)}")
for i, c in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(c.page_content)
    print("-----")


# 6. Create / populate the vector store in Postgres (pgvector)
vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embedding_model,
    connection=connection,
    use_jsonb=True,
)


# 7. Create retriever (fetch up to 10 docs)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})


# 8. Define the query
query = "AI engineer experience in Generative AI"


# 9. Retrieve top-k docs and deduplicate them
raw_docs = retriever.invoke(query)

retrieved_docs = []
seen = set()
for d in raw_docs:
    text = d.page_content.strip()
    if text not in seen:
        seen.add(text)
        retrieved_docs.append(d)

print("\nRetrieved documents (deduplicated):")
for doc in retrieved_docs:
    print(doc.page_content)
    print("---------------")


# 10. Build prompt template (kept for future LLM use)
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant that answers the details based on the context provided.

Context:
{context}

Question:
{question}
"""
)


# 11. Simple RAG-style answer WITHOUT an LLM (no APIs needed)
def naive_rag_answer(question, retrieved, all_chunks):
    print("\n--- Naive RAG-style Answer (no LLM, just stitched context) ---")
    print(f"Question: {question}\n")

    # Try to extract a clean "experience" sentence
    experience_lines = []

    # Search in retrieved docs first
    for d in retrieved:
        for line in d.page_content.splitlines():
            line = line.strip()
            if "AI Engineer" in line or "experience" in line:
                experience_lines.append(line)

    # If not found in retrieved docs, search all chunks
    if not experience_lines:
        for d in all_chunks:
            for line in d.page_content.splitlines():
                line = line.strip()
                if "AI Engineer" in line or "experience" in line:
                    experience_lines.append(line)

    experience_lines = list(dict.fromkeys(experience_lines))  # dedupe, preserve order

    if experience_lines:
        print("Extracted experience detail:")
        print(experience_lines[0])
    else:
        print("Could not find a specific experience line, showing best context instead.")

    # Also show stitched context from retrieved docs
    combined = "\n\n".join(d.page_content for d in retrieved)
    print("\nContext used:")
    print(combined)


# 12. Invoke the naive answer
naive_rag_answer(query, retrieved_docs, chunks)
