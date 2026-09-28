"""Runs benchmark queries against the running API.  Start the server first."""
import requests

URL = "http://127.0.0.1:8000/chat"
QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",  # out-of-scope: should refuse
]

for q in QUERIES:
    r = requests.post(URL, json={"query": q}, timeout=60).json()
    print("=" * 80)
    print("Q:", q)
    print("A:", r["answer"])
    print("Confidence:", r["confidence_score"], "| chunks:", len(r["retrieved_chunks"]))