# ---------------------------------------------------------
# Load and parse a PDF into LangChain Document objects
# ---------------------------------------------------------

# Import required modules
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob

# Create a Blob object from your PDF file path
# Replace with your own PDF filename
blob = Blob.from_path("./Arjun_Varma_Generative_AI_Resume.pdf")

# Initialize the PDF parser
# PyPDFParser can extract both text and metadata from PDFs
parser = PyPDFParser()

# Lazily parse the blob (generator-based for memory efficiency)
documents = parser.lazy_parse(blob)

# Store parsed Document objects in a list
docs = []

# Iterate through parsed pages (each page = one Document)
for doc in documents:
    docs.append(doc)

# Optional: inspect the first page’s content
# print(docs[0].page_content)

# Optional: inspect the first page’s metadata (like page number)
# print(docs[0].metadata)

# Print summary of what was loaded
print(f"✅ PDF successfully loaded — Total pages parsed: {len(docs)}")
print("Example metadata of first page:")
print(docs[0].metadata)
print("\nExample text preview:")
print(docs[0].page_content[:500])  # show first 500 characters
