# 🚀 LangChain, LangGraph, RAG, React, LLM Server - Full Project Repository

Welcome to the **AI & LLM Projects Repository**, containing hands‑on implementations of:

* **LangChain** (Prompting, Chains, Agents)
* **LangGraph** (stateful LLM workflows)
* **RAG Pipelines** (with pgvector + Docker)
* **React-based AI Apps**
* **LLM Server / LangServe APIs**

This repository is organised so you can learn, experiment, and build production-ready AI workflows.

---

## 📂 Project Structure

```
LangChain/
│
├── 1_introduction.py
├── 2_Prompt.py
├── 3_prompt-template.py
├── ... (basic LangChain functionality)
│
├── LANGRAPH/
│   ├── 1_simple_chatbot.py
│   ├── 2_chatbot_with_tools.py
│   ├── 3_chatbot_with_memory.py
│   ├── 4_Human_in_loop.py
│   ├── 6_rag_powered_tool_calling.ipynb
│   ├── requirements.txt
│
├── RAG/
│   ├── 1_data_loader_txt.py
│   ├── 2-pdf-loader.py
│   ├── 3-chunking-demo.py
│   ├── 4-embeddings-demo.py
│   ├── 5-data-ingestion-demo.py
│   ├── 6-vector-store-demo.py
│   ├── 7-rag-simple-demo.py
│   ├── 8-rag-chain.py
│   ├── 9_a_basic_part_1.py
│   ├── 10_b_basic_part_2.py
│   ├── 11_a_rag_basic_metadata.py
│   ├── 12b_rag_basic_metadata.py
│   ├── 13-vector-store-demo.py
│   ├── a14_rag_simple_demo.py (HuggingFace embeddings + pgvector)
│   ├── Arjun_Varma_Generative_AI_Resume.pdf
│   │
│   └── ReAct/
│       ├── 1_basic_with_own_Prompt.py
│       ├── 2_basics.py
│       ├── 3_websearch.py
│       ├── requirements.txt
│
└── README.md
```

---

## 🧠 Technologies Used

### **LLM Frameworks**

* **LangChain** – chains, prompts, agents, retrievers
* **LangGraph** – state machines for complex conversational workflows
* **LangServe** (optional) – deploy LLM apps as APIs

### **RAG (Retrieval-Augmented Generation)**

* **pgvector** vector store
* **PostgreSQL** running in **Docker**
* **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)
* **PDF parsing**, metadata, chunking, similarity search

### **Front‑End**

* React (if included later)

### **Python Stack**

* Python 3.10+
* dotenv
* langchain-core
* langchain-community
* langchain-huggingface
* langchain-postgres
* sentence-transformers

---

## 🐳 How to Run the RAG System (pgvector + Docker)

### **1️⃣ Start PostgreSQL + pgvector**

Run this once:

```bash
docker run --name pgvector-container \
  -e POSTGRES_USER=langchain \
  -e POSTGRES_PASSWORD=langchain \
  -e POSTGRES_DB=langchain \
  -p 6024:5432 -d pgvector/pgvector:pg16
```

Start it anytime:

```bash
docker start pgvector-container
```

---

## 🧪 Run a RAG Script

Example:

```bash
python RAG/a14_rag_simple_demo.py
```

This demo includes:

* HuggingFace embeddings (no API key needed)
* PDF parsing
* Chunking
* pgvector indexing
* Top‑K similarity search

---

## 🧱 Running LangGraph Examples

Go into the folder:

```bash
cd LANGRAPH
python 1_simple_chatbot.py
```

---

## 🧑‍💻 Recommended `.gitignore`

Your repo already excludes sensitive & large files:

```
.venv/
.env
LANGRAPH/.env
__pycache__/
*.pyc
.DS_Store
```

---

## ⚠️ Security Notice

This repository previously contained an `.env` file with an exposed OpenAI key.
It has now been removed and cleaned from git history.
Always keep API keys **out of Git**.

---

## ⭐ Future Enhancements

* Add frontend UI for RAG (React)
* Add LangServe API endpoints
* Add Docker Compose for full stack
* Add notebook tutorials

---

## 🤝 Contributing

Pull requests are welcome! If you find bugs or want to add examples, feel free to open an issue.

---

## 👨‍💻 Author

**Param Purohit**

* 📧 Email: [purohit.param91@gmail.com](mailto:purohit.param91@gmail.com)
* 🔗 LinkedIn: [https://www.linkedin.com/in/param-p-370616310/](https://www.linkedin.com/in/param-p-370616310/)
* 🧳 Portfolio/Data Science Repo: [https://github.com/Purohit1999/Data_Science](https://github.com/Purohit1999/Data_Science)

---

If you'd like, I can add:
✅ badges (Python version, Docker, pgvector, HF models)
✅ screenshots
✅ a quickstart guide

Just tell me! 🚀
