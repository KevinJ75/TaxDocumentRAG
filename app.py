"""
app.py — IRS Tax Document Assistant (LangChain version).

Features:
- Conversational chat interface with memory across follow-up questions
- Streaming responses word-by-word
- Source passage viewer per answer
- Persistent ChromaDB vector store

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CHROMA_PATH = "./chroma_db_lc"
COLLECTION_NAME = "irs_documents"
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Cached resources — loaded once per session
# ---------------------------------------------------------------------------

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


@st.cache_resource
def load_llm():
    return ChatAnthropic(
        model=MODEL,
        api_key=ANTHROPIC_API_KEY,
        streaming=True,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_docs(docs):
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_messages(question, chat_history, context):
    """Build the full message list for Claude, including conversation history."""
    system_content = f"""You are a tax expert assistant helping an accountant navigate IRS documents.

Based on the following excerpts from IRS publications, answer the question accurately and concisely.
Always cite your sources using the format (Source: [document name], Page [X]).
If the context does not contain enough information to answer, say so clearly.
When answering follow-up questions, use the conversation history for context.

CONTEXT:
{context}"""

    messages = [SystemMessage(content=system_content)]
    for human_msg, ai_msg in chat_history:
        messages.append(HumanMessage(content=human_msg))
        messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=question))
    return messages


def stream_response(question, chat_history, context, llm):
    """Generator that yields Claude's response tokens one at a time."""
    messages = build_messages(question, chat_history, context)
    for chunk in llm.stream(messages):
        yield chunk.content


def get_chat_history():
    """Extract (question, answer) pairs from session messages for the LLM context."""
    history = []
    msgs = st.session_state.get("messages", [])
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            history.append((msgs[i]["content"], msgs[i + 1]["content"]))
    return history


def run_ingestion(force=False):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    import ingest
    ingest.PDF_DIR = project_dir
    ingest.CHROMA_PATH = os.path.join(project_dir, "chroma_db_lc")
    ingest.ingest_pdfs(force=force)


def render_sources(sources):
    with st.expander("View source passages"):
        for i, src in enumerate(sources):
            st.markdown(
                f"**{i + 1}. {src['source']} — Page {src['page']} (Tax Year: {src['tax_year']})**"
            )
            text = src["text"]
            st.text(text[:600] + "..." if len(text) > 600 else text)
            st.divider()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IRS Tax Document Assistant",
    page_icon="📋",
    layout="wide",
)
st.title("📋 IRS Tax Document Assistant")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Document Database")

    db_ready = os.path.exists(CHROMA_PATH)
    vectorstore = None

    if db_ready:
        try:
            vectorstore = load_vectorstore()
            count = vectorstore._collection.count()
            st.success(f"{count:,} chunks indexed")

            meta = vectorstore._collection.get(include=["metadatas"])
            sources = sorted(set(m["source"] for m in meta["metadatas"]))
            st.markdown("**Loaded documents:**")
            for s in sources:
                st.markdown(f"- {s}")
        except Exception as e:
            st.warning(f"Could not load database: {e}")
            vectorstore = None
    else:
        st.info("No database found. Ingest your PDFs to get started.")

    st.divider()

    if vectorstore is None:
        if st.button("Ingest Documents", type="primary", use_container_width=True):
            with st.spinner("Processing PDFs... (may take a minute on first run)"):
                try:
                    run_ingestion(force=False)
                    st.cache_resource.clear()
                    st.success("Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
    else:
        if st.button("Re-ingest Documents", type="secondary", use_container_width=True):
            with st.spinner("Re-processing PDFs..."):
                try:
                    run_ingestion(force=True)
                    st.cache_resource.clear()
                    st.success("Done!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Re-ingestion failed: {e}")

    st.divider()
    n_results = st.slider("Passages to retrieve", min_value=3, max_value=10, value=5)

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
if vectorstore is not None:
    llm = load_llm()

    # Render existing conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    # Chat input — always pinned to the bottom
    if prompt := st.chat_input("Ask a tax question..."):

        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Retrieve relevant passages and stream answer
        with st.chat_message("assistant"):
            docs = vectorstore.similarity_search(prompt, k=n_results)
            context = format_docs(docs)
            chat_history = get_chat_history()

            full_response = st.write_stream(
                stream_response(prompt, chat_history, context, llm)
            )

            source_data = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "tax_year": doc.metadata.get("tax_year", "unknown"),
                    "text": doc.page_content,
                }
                for doc in docs
            ]
            render_sources(source_data)

        # Store assistant message with sources for history rendering
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": source_data,
        })

else:
    st.info("Use the **Ingest Documents** button in the sidebar to index your IRS PDFs, then ask away.")
