# 1_data_loader_txt.py
# Load a .txt file as LangChain Documents and print a quick summary.

import os
import sys
from typing import List

# Works across recent LangChain versions
try:
    from langchain_community.document_loaders import TextLoader
except Exception as e:
    print("Install dependencies first: pip install langchain langchain-community")
    raise

def load_txt(path: str, encoding: str = "utf-8") -> List:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    loader = TextLoader(path, encoding=encoding)
    return loader.load()   # -> List[Document]

def preview(doc, n=400) -> str:
    text = doc.page_content.replace("\n", " ").strip()
    return (text[:n] + "…") if len(text) > n else text

if __name__ == "__main__":
    # Usage: python 1_data_loader_txt.py agentic_ai_sample.txt
    file_path = sys.argv[1] if len(sys.argv) > 1 else "agentic_ai_sample.txt"

    docs = load_txt(file_path, encoding="utf-8")

    print(f"\n✅ Number of documents loaded: {len(docs)}")
    for i, d in enumerate(docs, start=1):
        meta = d.metadata if hasattr(d, "metadata") else {}
        print(f"\n--- Document {i} ---")
        print(f"Source : {meta.get('source', file_path)}")
        print(f"Chars  : {len(d.page_content)}")
        print(f"Preview: {preview(d, 300)}")
