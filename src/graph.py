"""LangGraph workflow: retrieve -> generate, with score-based grounding."""
from typing import List, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from src import config


class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


def build_rag_graph():
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=config.PINECONE_INDEX_NAME, embedding=embeddings
    )
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)

    def retrieve_node(state: AgentState):
        # Pinecone cosine similarity: higher = more relevant
        results = vectorstore.similarity_search_with_score(
            state["question"], k=config.TOP_K
        )
        contexts = [doc.page_content for doc, _ in results]
        top_score = max((s for _, s in results), default=0.0)
        return {"context": contexts, "score": round(float(top_score), 4)}

    def generate_node(state: AgentState):
        # Out-of-scope guard: weak retrieval => refuse without calling the LLM
        if not state["context"] or state["score"] < config.RELEVANCE_THRESHOLD:
            return {"answer": config.REFUSAL_MESSAGE}

        context_str = "\n\n---\n\n".join(state["context"])
        prompt = f"""You are a strict assistant. Answer the question relying ONLY on the context below.
Do not use outside knowledge. If the context does not contain enough information,
reply exactly: '{config.REFUSAL_MESSAGE}'

Context:
{context_str}

Question: {state['question']}"""
        response = llm.invoke(prompt)
        answer = response.content.strip()
        # If the model refused, confidence in an answer is zero
        if config.REFUSAL_MESSAGE.lower() in answer.lower():
            return {"answer": answer, "score": 0.0}
        return {"answer": answer}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()