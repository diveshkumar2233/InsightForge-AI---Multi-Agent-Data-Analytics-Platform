"""
agents/rag_agent.py
────────────────────
RAG pipeline:
  • Ingest PDF / text documents → chunk → embed → store in ChromaDB
  • Semantic search with source citations
  • Conversational Q&A over retrieved context via Groq LLM
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Optional

import pdfplumber
from loguru import logger

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
)
from agents.supervisor import AgentState


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Fixed-size character chunker with overlap."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ── PDF extractor ─────────────────────────────────────────────────────────────

def extract_pdf_text(data: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text


# ── ChromaDB client (lazy init) ───────────────────────────────────────────────

_chroma_client  = None
_chroma_coll    = None


def _get_collection():
    global _chroma_client, _chroma_coll
    if _chroma_coll is not None:
        return _chroma_coll

    try:
        import chromadb
        from chromadb.config import Settings

        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_coll = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"[RAG] ChromaDB collection '{CHROMA_COLLECTION}' ready")
    except Exception as e:
        logger.error(f"[RAG] ChromaDB init failed: {e}")
        raise
    return _chroma_coll


# ── Embedding (Google) ────────────────────────────────────────────────────────

def _embed(texts: list[str]) -> list[list[float]]:
    """Embed using Google Generative AI embeddings."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    result = genai.embed_content(
        model="models/embedding-001",
        content=texts,
        task_type="retrieval_document",
    )
    return result["embedding"] if isinstance(result["embedding"][0], list) else [result["embedding"]]


# ── Ingest ────────────────────────────────────────────────────────────────────

def ingest_document(name: str, text: str) -> int:
    """Chunk, embed, and store a document. Returns number of chunks stored."""
    collection = _get_collection()
    chunks     = chunk_text(text)
    if not chunks:
        logger.warning(f"[RAG] No text extracted from {name}")
        return 0

    embeddings = _embed(chunks)
    doc_hash   = hashlib.md5(text.encode()).hexdigest()[:8]
    ids        = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas  = [{"source": name, "chunk": i} for i in range(len(chunks))]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    logger.success(f"[RAG] Ingested {len(chunks)} chunks from '{name}'")
    return len(chunks)


def ingest_pdf_bytes(name: str, data: bytes) -> int:
    text = extract_pdf_text(data)
    return ingest_document(name, text)


# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search. Returns list of {text, source, score}."""
    collection = _get_collection()
    q_emb      = _embed([query])
    results    = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text"  : doc,
            "source": meta.get("source", "unknown"),
            "chunk" : meta.get("chunk", 0),
            "score" : round(1 - dist, 4),   # cosine similarity
        })
    return hits


# ── RAG Q&A ───────────────────────────────────────────────────────────────────

def rag_answer(query: str, top_k: int = 5, chat_history: list = None) -> dict:
    """Full RAG pipeline: retrieve + generate answer with citations."""
    if not GROQ_API_KEY:
        return {"answer": "GROQ_API_KEY not set.", "sources": []}

    hits = retrieve(query, top_k=top_k)
    if not hits:
        return {"answer": "No relevant documents found. Please upload documents first.", "sources": []}

    context = "\n\n".join(
        f"[Source: {h['source']} | Chunk {h['chunk']} | Score: {h['score']}]\n{h['text']}"
        for h in hits
    )

    sources_block = "\n".join(
        f"- **{h['source']}** (chunk {h['chunk']}, relevance {h['score']})"
        for h in hits
    )

    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=GROQ_TEMPERATURE)

    messages = [
        SystemMessage(content=(
            "You are a precise document analyst. Answer ONLY based on the provided context. "
            "If the answer is not in the context, say so clearly. "
            "Always cite the source document name in your answer."
        )),
    ]
    if chat_history:
        messages.extend(chat_history[-6:])   # last 3 turns

    messages.append(HumanMessage(content=(
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed, well-structured answer with inline citations like [Source: filename]."
    )))

    response = llm.invoke(messages)
    answer   = response.content

    return {
        "answer" : answer,
        "sources": sources_block,
        "hits"   : hits,
    }


def collection_stats() -> dict:
    try:
        coll = _get_collection()
        return {"count": coll.count(), "name": CHROMA_COLLECTION}
    except Exception as e:
        return {"error": str(e)}


# ── LangGraph Node ────────────────────────────────────────────────────────────

def rag_node(state: AgentState) -> AgentState:
    query   = state.get("user_query", "")
    history = [m for m in state.get("messages", [])[-6:]]

    try:
        result  = rag_answer(query, chat_history=history)
        answer  = result["answer"]
        sources = result.get("sources", "")
        if sources:
            answer += f"\n\n**Sources:**\n{sources}"
        return {**state, "rag_result": result, "final_answer": answer}
    except Exception as e:
        logger.exception(f"[RAG] node error: {e}")
        return {**state, "error": str(e)}
