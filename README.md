# IRS Tax Document RAG

A retrieval-augmented generation (RAG) assistant that lets users ask natural-language questions against IRS publications and get cited, source-grounded answers. Built to reduce the administrative overhead of navigating dense tax PDFs.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Claude](https://img.shields.io/badge/Claude-Sonnet%204.6-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)

---

## Why this project

Tax professionals routinely cross-reference hundreds of pages of IRS guidance to answer client questions. This tool turns that lookup into a conversation: you ask, the model retrieves the relevant passages, and you get an answer with the exact publication and page number it came from.

The architecture is intentionally generic: swap the IRS PDFs for any document base (Confluence exports, legal contracts, internal wikis) and the same pipeline applies.

## Features

- **Conversational chat interface** with multi-turn memory — follow-up questions retain context
- **Streaming responses** rendered token-by-token for a responsive UX
- **Source citations** on every answer, linked to a viewer that shows the exact retrieved passages
- **Persistent vector store** (ChromaDB) — ingest once, query forever
- **Year-aware metadata** — chunks are tagged with their tax year so the model can disambiguate across publications
- **One-click re-ingestion** from the sidebar when documents are added or updated

## Architecture

```
PDFs ──▶ PDFPlumber ──▶ RecursiveCharacterTextSplitter ──▶ all-MiniLM-L6-v2
                                                                 │
                                                                 ▼
User ──▶ Streamlit chat ──▶ Chroma similarity search ──▶ ChromaDB (persisted)
                                    │
                                    ▼
                            Top-k passages + chat history
                                    │
                                    ▼
                       Claude Sonnet 4.6 (streaming) ──▶ Cited answer
```

**Ingestion pipeline** (`ingest.py`): PDFs are parsed with PDFPlumber, split into 800-character chunks with 100-character overlap, embedded with `all-MiniLM-L6-v2` (384-dim sentence-transformer), and persisted to ChromaDB. Each chunk carries `source`, `page`, and `tax_year` metadata.

**Query pipeline** (`app.py`): The user's question triggers a top-k similarity search against the vector store. Retrieved passages plus prior conversation turns are formatted into a system prompt that instructs Claude to cite sources in `(Source: [doc], Page [X])` format. The response streams back via LangChain's `ChatAnthropic.stream()`.

## Tech stack

| Layer            | Choice                              | Rationale                                         |
| ---------------- | ----------------------------------- | ------------------------------------------------- |
| LLM              | Claude Sonnet 4.6 (`langchain-anthropic`) | Strong instruction-following and citation discipline |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2` | Local, free, fast — no embedding API calls       |
| Vector store     | ChromaDB (persistent)               | Zero-config local deployment, LangChain-native    |
| PDF parsing      | PDFPlumber                          | Robust text extraction with page-level metadata   |
| Orchestration    | LangChain 0.3                       | Composable retrievers, splitters, chat models     |
| UI               | Streamlit                           | Stateful chat with `st.chat_message` and `st.write_stream` |

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/KevinJ75/TaxDocumentRAG.git
cd TaxDocumentRAG

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example config.env
# Edit config.env and set ANTHROPIC_API_KEY=...
```

Place any IRS PDFs you want to query in the project root (the repo ships with `2025_Instruction1040.pdf` and `2026_Pub15.pdf` as examples).

## Usage

```bash
# Build the vector store (one-time, or after adding new PDFs)
python ingest.py

# Force a clean rebuild
python ingest.py --force

# Launch the chat UI
streamlit run app.py
```

The first run will index the PDFs in the current directory. Subsequent launches reuse the persisted store. Use the **Re-ingest Documents** button in the sidebar to refresh after adding new files.

### Example interaction

> **You:** What's the standard deduction for a single filer in 2025?
>
> **Assistant:** For tax year 2025, the standard deduction for a single filer is $15,000 *(Source: 2025_Instruction1040.pdf, Page 34)*.
>
> **You:** And for someone over 65?
>
> **Assistant:** A single filer who is 65 or older receives an additional $2,000, bringing the total to $17,000 *(Source: 2025_Instruction1040.pdf, Page 34)*.

## Project structure

```
.
├── app.py              # Streamlit chat UI + retrieval/generation loop
├── ingest.py           # PDF parsing, chunking, embedding, persistence
├── requirements.txt
├── .env.example        # Template — copy to config.env and add your key
├── chroma_db_lc/       # Persisted vector store (gitignored)
└── *.pdf               # IRS source documents
```

## Design decisions

- **Local embeddings over a hosted API.** `all-MiniLM-L6-v2` runs on CPU in seconds and keeps document content on-device — no per-token cost and no data egress. For a corpus of legal/financial documents this is a meaningful default.
- **Chunk size of 800 with 100 overlap.** Tax publications use long, structured paragraphs; smaller chunks fragment context, larger chunks dilute relevance. The overlap preserves continuity across boundaries.
- **Caching with `@st.cache_resource`.** The embedding model and vector store are loaded once per session, not on every query — Streamlit otherwise reloads on each rerun.
- **Conversation history threaded into the prompt.** Rather than relying on a stateful chain, the chat history is reconstructed from `st.session_state` and passed as alternating `HumanMessage`/`AIMessage` objects, keeping the retrieval and generation steps decoupled.
