# 🔍 Retrieval-Augmented Generation (RAG) Pipeline

This repository hosts a **graph-enhanced RAG pipeline** designed for advanced code reasoning and knowledge retrieval. It supports two intelligent assistant modes:

1. **General Chatbot Mode** – Answers queries using external web sources (Arxiv + Tavily).
2. **Codebase Assistant Mode** – Expertly responds to questions about the [Bitcoindev mailing list](https://github.com/bitcoin/bitcoin-dev) using:
   - A **static FAISS index** over emails and source code
   - A **graph-based structure** built from class/function nodes and their relationships

It incorporates cutting-edge components like:
- **Groq + Llama 3/4** for blazing-fast query rewriting and code generation
- **Together API** for embedding generation and LLM support
- **LangChain + FAISS** for modular document retrieval
- **Graph Reasoning Engine** that encodes interlinked code structure

---

## 📁 Directory Overview

```bash
project_root/
├── backend/
│   ├── general/                  # General chatbot pipeline
│   │   ├── general_rag.py
│   │   ├── index.py
│   │   ├── ingestion.py
│   │   ├── ingestion_git.py
│   │   └── prompts.py
│   ├── codebase/                 # Codebase assistant pipeline
│   │   ├── codebase_rag.py
│   │   ├── utils.py              # Graph utilities
│   │   ├── prompt.py             # Prompt loader
├── config/                       # Environment config
│   └── settings.py
├── embeddings/                   # Live embeddings cache
├── faiss_index/                  # Static FAISS index storage
├── bitcoindev.git/               # Git mirror of BitcoinDev mailing list
├── email_extract.py              # Prepares data for ingestion
├── .env                          # API keys and config
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## ✨ Key Features

### 🚦 Dual Modes of Operation
- **General Assistant** – Pulls live results from Arxiv and Tavily, ideal for research.
- **Codebase Assistant** – Answers contextually from a Git snapshot of the BitcoinDev mailing list.

### 🧭 Graph Reasoning Engine
- Builds a **directed graph** where:
  - **Nodes** are `classes`, `functions`, or `files`
  - **Edges** represent function calls, inheritance, imports, or nesting
- Used for **semantic expansion** of retrieval scope
- Contextual neighborhood of query-relevant nodes is embedded and retrieved

### 🧠 LLM-Orchestrated RAG
- **Query Rewriting** via Groq’s Llama 4
- **Context Building** using graph traversal + FAISS hybrid
- **Answer Synthesis** with code-aware prompting

### 🧾 Modular Prompting System
- All prompts are defined in `prompts.py`
- Includes:
  - Query rewriter system message
  - Graph reasoning assistant prompt
  - Code generation prompt
- Integrated with Pydantic for structured outputs

### 🛠 Together + Groq Integration
- **Together AI** used for sentence-transformers embeddings and fallback LLMs
- **Groq API** enables low-latency completions using Llama 3/4

---

## 🧠 How It Works

### 1. Ingestion Pipeline
- `ingestion.py`: Uses Arxiv + Tavily APIs to collect content
- `ingestion_git.py`: Converts email threads and code snippets from the Git archive
- `email_extract.py`: Prepares emails in JSONL format

### 2. Index
- `index.py`: Builds FAISS index for both dynamic and static content

### 3. Agent Flow (codebase_rag.py)
1. Query Rewriting using `Groq` LLM and rewrite prompt
2. Retrieval using:
   - Graph traversal of neighboring code units
3. Answer generation using `code_reasoning_prompt`
  
### 4. Agent Flow (general_rag.py)
1. Query Rewriting using `Groq` LLM and rewrite prompt
2. Retrieval using:
   - Faiss Index
3. Implemeted `Self-Rag` to resolve the problem of `hallucinations`

### 5. Prompts
- Loaded from `prompts.py`
- Output validated with `pydantic` schemas like `QueryReWrite`

---

## ⚙️ Setup Instructions

### Step 1: Clone & Setup
```bash
git clone https://github.com/naitik-2006/web3_rag_pipeline
cd project_root
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment
Create a `.env` file in the root with:
```bash
TOGETHER_API_KEY=your-key
TAVILY_API_KEY=your-key
GROQ_API_KEY=your-key

EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
VECTORSTORE_TYPE=faiss
STATIC_VS_SAVE_PATH=./embeddings/static_index
```

---

## 🚀 How To Run

### 1. Ingest Knowledge
```bash
python -m backend.general.ingestion         # Dynamic: Arxiv + Tavily
python -m backend.general.ingestion_git     # Static: Mailing list
```

### 2. Build Index
```bash
python -m backend.general.index
```

### 3. Run Agent for Answering
```bash
# For running simple chabot
python -m backend.general.general_rag.py

# For question tailored to your codebase
python -m backend.codebase.codebase_rag <path-code-base>
```

This executes the full pipeline:
- Query rewrite → Contextual retrieval → Code reasoning → Answer generation

---

## 🙋 FAQ

**Q: Can I replace the LLMs with OpenAI or Anthropic?**
> Yes, swap the `LLM` wrapper in `backend` and update your `.env`

**Q: Can I plug in my own repo/codebase?**
> Yes! Replace the `bitcoindev.git` repo and rerun `ingestion_git.py`

---

## 🧑‍💻 Maintainer
Built with ❤️ by [Naitik Agrawal]. Contributions and feedback are always welcome.
