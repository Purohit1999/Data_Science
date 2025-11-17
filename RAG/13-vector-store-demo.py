from dotenv import load_dotenv

from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load env vars (not required for HF embeddings, but handy for future use)
load_dotenv()

# 2. Postgres / pgvector connection (Docker)
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# 3. Embedding model (FREE, no OpenAI quota needed)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Load PDF
blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")
parser = PyPDFParser()

documents = parser.lazy_parse(blob)
docs = [doc for doc in documents]

print(docs)

# 5. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,   # Adjust as needed
    chunk_overlap=50  # Adjust as needed
)

chunks = splitter.split_documents(docs)
print(f"Number of chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk.page_content)
    print("--------------------------------")

# 6. Create vector store in Postgres
vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embedding_model,
    connection=connection,
    use_jsonb=True,   # optional, but recommended for metadata
)

print("Vector Store created:")
print(vector_store)

# 7. Query the vector store
query = "AI engineer"

# Top 3 most similar docs
results = vector_store.similarity_search(query, k=3)
print(f"Results for query '{query}':")
for i, result in enumerate(results, start=1):
    print(f"Result {i}:")
    print(f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print("--------------------------------")

# Same query, but with similarity scores
results_with_scores = vector_store.similarity_search_with_score(query, k=3)
print(f"Results with scores for query '{query}':")
for i, (result, score) in enumerate(results_with_scores, start=1):
    print(f"Result {i}:")
    print(f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print(f"Score: {score}")
    print("--------------------------------")


# 8. Simple "answer" generator without an LLM: just stitches best chunks
def naive_answer(question: str, docs):
    print("\n--- Naive Answer (no LLM, just stitched context) ---")
    combined = " ".join(d.page_content for d in docs)
    print(f"Question: {question}")
    print("\nAnswer context:")
    print(combined)


# Use the top `results` as the context for the naive answer
naive_answer(query, results)
