"""PDF loading, chunking, embedding and Pinecone index setup.

Run once:  python -m src.ingestion
"""
import os
import requests
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


def download_pdf() -> None:
    if os.path.exists(config.PDF_PATH):
        print(f"PDF already at {config.PDF_PATH}")
        return
    os.makedirs(os.path.dirname(config.PDF_PATH), exist_ok=True)
    resp = requests.get(config.PDF_URL, timeout=60)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise RuntimeError(
            "Download did not return a PDF (Drive may need manual download). "
            f"Save the file manually as {config.PDF_PATH}."
        )
    with open(config.PDF_PATH, "wb") as f:
        f.write(resp.content)
    print("PDF downloaded.")


def ensure_index() -> None:
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    existing = [i["name"] for i in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created index {config.PINECONE_INDEX_NAME}")


def run_ingestion() -> PineconeVectorStore:
    download_pdf()
    ensure_index()

    docs = PyPDFLoader(config.PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    store = PineconeVectorStore.from_documents(
        documents=chunks, embedding=embeddings, index_name=config.PINECONE_INDEX_NAME
    )
    print("Upserted chunks to Pinecone.")
    return store


if __name__ == "__main__":
    run_ingestion()