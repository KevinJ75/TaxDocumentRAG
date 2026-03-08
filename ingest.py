"""
ingest.py — Parse IRS PDFs and build the ChromaDB vector store using LangChain.

Run once to build the store, then re-run with --force to rebuild from scratch.

Usage:
    python ingest.py
    python ingest.py --force
"""

import os
import re
import sys
import shutil

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db_lc"
PDF_DIR = "."
COLLECTION_NAME = "irs_documents"


def ingest_pdfs(force=False):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Skip if already ingested
    if os.path.exists(CHROMA_PATH) and not force:
        vs = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        count = vs._collection.count()
        if count > 0:
            print(f"Already ingested {count} chunks. Use --force to re-ingest.")
            return

    if force and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared existing vector store.")

    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdfs:
        print("No PDF files found in current directory.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )

    all_docs = []
    for pdf_file in pdfs:
        year_match = re.search(r"(\d{4})", pdf_file)
        tax_year = year_match.group(1) if year_match else "unknown"

        print(f"Loading {pdf_file} (tax year: {tax_year})...")
        loader = PDFPlumberLoader(os.path.join(PDF_DIR, pdf_file))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_file
            page.metadata["page"] = int(page.metadata.get("page", 0)) + 1  # convert to 1-indexed
            page.metadata["tax_year"] = tax_year

        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)
        print(f"  -> {len(chunks)} chunks from {len(pages)} pages")

    print(f"\nEmbedding and storing {len(all_docs)} total chunks...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    print(f"Done. Vector store saved to {CHROMA_PATH}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    ingest_pdfs(force=force)
