"""Environment setup and constants shared across the project."""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PDF_URL = "https://drive.google.com/uc?export=download&id=15VLphKcY23_fpYxN62UEQRri_psRVfP9"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Cosine similarity of the best chunk below this => treat question as out-of-scope
RELEVANCE_THRESHOLD = 0.30

REFUSAL_MESSAGE = "I cannot answer based on the provided document."

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY in your .env file.")