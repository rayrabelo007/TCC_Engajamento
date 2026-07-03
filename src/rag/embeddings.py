# ==========================================
# RAG - GERAÇÃO DE EMBEDDINGS
# ==========================================
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Modelo multilíngue leve, com bom suporte a Português
DEFAULT_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Carrega um modelo SentenceTransformer para geração de embeddings."""
    return SentenceTransformer(model_name)


def generate_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """Gera embeddings densos para uma lista de textos usando o modelo fornecido."""
    return model.encode(list(texts), show_progress_bar=True)
