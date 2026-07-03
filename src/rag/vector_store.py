# ==========================================
# RAG - VECTOR STORE (FAISS)
# ==========================================
from typing import Tuple

import numpy as np
import faiss


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Constrói um índice FAISS (L2) a partir de uma matriz de embeddings."""
    embeddings = np.asarray(embeddings, dtype='float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Busca os top_k vizinhos mais próximos de um embedding de consulta no índice."""
    query_embedding = np.asarray(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]


def save_index(index: faiss.Index, path: str) -> None:
    """Salva um índice FAISS em disco."""
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    """Carrega um índice FAISS salvo em disco."""
    return faiss.read_index(path)
