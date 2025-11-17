# 3-chunking-demo.py
# pip install -U langchain langchain-community pypdf

from pathlib import Path
from langchain_community.document_loaders.parsers.pdf import PyPDFParser
from langchain_core.documents.base import Blob
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_as_docs(pdf_path: str):
    """Parse a PDF into LangChain Documents using PyPDFParser."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")

    blob = Blob.from_path(str(path))
    parser = PyPDFParser()

    # lazy_parse returns an iterator; materialize to a list
    docs = [doc for doc in parser.lazy_parse(blob)]
    return docs


def chunk_docs(
    docs,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    add_start_index: bool = True,
):
    """Split Documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,  # helpful metadata field
    )
    return splitter.split_documents(docs)


def main():
    # ---- change this to your file path ----
    pdf_path = "./Arjun_Varma_Generative_AI_Resume.pdf"
    # ---------------------------------------

    print(f"Loading: {pdf_path}")
    docs = load_pdf_as_docs(pdf_path)
    print(f"Parsed {len(docs)} document(s) from the PDF.")

    # Show a peek at parsed metadata
    if docs:
        print("Sample document metadata:", docs[0].metadata)

    # Split
    chunks = chunk_docs(docs, chunk_size=1000, chunk_overlap=100)
    print(f"\nNumber of chunks created: {len(chunks)}\n")

    # Print chunk previews
    for i, chunk in enumerate(chunks, start=1):
        header = f"--- Chunk {i} (len={len(chunk.page_content)} chars) ---"
        print(header)
        print(chunk.page_content.strip())
        print("-" * len(header))

    # If you only want to inspect the first few chunks, comment the loop above
    # and uncomment the snippet below:
    # for i, chunk in enumerate(chunks[:5], start=1):
    #     print(f"Chunk {i}:\n{chunk.page_content}\n{'-'*40}")


if __name__ == "__main__":
    main()
