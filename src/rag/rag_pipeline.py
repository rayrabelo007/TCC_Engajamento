# ==========================================
# RAG - PIPELINE DE INDEXAÇÃO E RECUPERAÇÃO
# ==========================================
from typing import Any, Dict

import pandas as pd

from src.rag.retriever import build_retriever


def run_rag_pipeline(df_comments: pd.DataFrame, text_column: str = 'cleaned_text') -> Dict[str, Any]:
    """Indexa os comentários limpos de df_comments para recuperação por similaridade (RAG)."""
    comments = df_comments[text_column].tolist()
    print("=> Indexando comentários para RAG (embeddings + FAISS)...")
    retriever = build_retriever(comments)
    print(f"[SUCESSO] Retriever RAG construído com {len(comments)} comentários indexados!")
    return retriever
