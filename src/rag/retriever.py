# ==========================================
# RAG - RETRIEVER
# ==========================================
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.rag.embeddings import load_embedding_model, generate_embeddings
from src.rag.vector_store import build_index, search


def build_retriever(comments: List[str], model: Optional[SentenceTransformer] = None) -> Dict[str, Any]:
    """Indexa uma lista de comentários (embeddings + FAISS) e retorna um retriever pronto para consultas."""
    model = model or load_embedding_model()
    comments = list(comments)
    embeddings = generate_embeddings(comments, model)
    index = build_index(embeddings)
    return {'comments': comments, 'model': model, 'index': index}


def retrieve_top_k(retriever: Dict[str, Any], query: str, top_k: int = 5) -> List[str]:
    """Retorna os top_k comentários mais similares semanticamente à query."""
    query_embedding = generate_embeddings([query], retriever['model'])[0]
    _, indices = search(retriever['index'], query_embedding, top_k=top_k)
    return [retriever['comments'][i] for i in indices if i != -1]
