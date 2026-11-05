# ==========================================
# RAG - RETRIEVER (COM SUPORTE A ESCOPO POR VÍDEO)
# ==========================================
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.rag.embeddings import load_embedding_model, generate_embeddings
from src.rag.vector_store import build_index, search


def build_retriever(
    comments: List[str],
    video_ids: Optional[List[str]] = None,
    model: Optional[SentenceTransformer] = None,
) -> Dict[str, Any]:
    """Indexa uma lista de comentários (embeddings + FAISS) com suporte opcional a video_id."""
    model = model or load_embedding_model()
    comments = list(comments)
    video_ids = list(video_ids) if video_ids is not None else ["N/A"] * len(comments)

    embeddings = generate_embeddings(comments, model)
    index = build_index(embeddings)
    
    return {
        'comments': comments,
        'video_ids': video_ids,
        'model': model,
        'index': index,
    }


def retrieve_top_k(
    retriever: Dict[str, Any],
    query: str,
    top_k: int = 5,
    target_video_id: Optional[str] = None,
) -> List[str]:
    """Retorna os top_k comentários mais similares, com opção de filtragem estrita por video_id."""
    search_k = top_k * 5 if target_video_id else top_k
    search_k = min(search_k, len(retriever['comments']))

    query_embedding = generate_embeddings([query], retriever['model'])[0]
    _, indices = search(retriever['index'], query_embedding, top_k=search_k)

    results = []
    video_ids = retriever.get('video_ids', ["N/A"] * len(retriever['comments']))

    for i in indices:
        if i == -1:
            continue
        v_id = video_ids[i]
        
        # Se um target_video_id foi passado, filtra estritamente por ele
        if target_video_id is None or v_id == target_video_id:
            results.append(retriever['comments'][i])
            if len(results) >= top_k:
                break

    return results