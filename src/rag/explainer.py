# ==========================================
# RAG - EXPLICAÇÃO DAS PREVISÕES (LLM)
# ==========================================
from typing import Any, Dict, List, Optional

import anthropic
import pandas as pd

from src import config
from src.modeling import FEATURE_COLUMNS
from src.rag.retriever import retrieve_top_k

SYSTEM_PROMPT: str = (
    "Você é um assistente que explica, de forma clara e objetiva, por que um "
    "modelo de machine learning previu determinado nível de engajamento para um "
    "vídeo, com base em métricas de sentimento dos comentários dos usuários. "
    "Baseie-se apenas nos dados fornecidos, evite jargão técnico desnecessário "
    "e responda sempre em português."
)


def build_feature_importance_summary(
    reg_model: Optional[Any], feature_cols: List[str] = FEATURE_COLUMNS
) -> Optional[str]:
    """Resume a importância das features de um modelo treinado em uma string legível."""
    if reg_model is None or not hasattr(reg_model, 'feature_importances_'):
        return None
    ranked = sorted(zip(feature_cols, reg_model.feature_importances_), key=lambda x: x[1], reverse=True)
    return ", ".join(f"{name} ({importance:.2f})" for name, importance in ranked)


def build_prediction_prompt(
    video_row: pd.Series,
    feature_importance_summary: Optional[str] = None,
    similar_comments: Optional[List[str]] = None,
) -> str:
    """Monta o prompt de usuário com as métricas do vídeo, importância das features e comentários similares."""
    lines = [
        f"Vídeo: {video_row.get('video_title', 'N/A')}",
        f"Taxa de engajamento prevista: {video_row.get('engagement_rate'):.4f}",
        f"Classe de engajamento: {'Alto' if video_row.get('engagement_class') == 1 else 'Baixo'}",
        f"Proporção de comentários positivos: {video_row.get('prop_positivo'):.2%}",
        f"Proporção de comentários negativos: {video_row.get('prop_negativo'):.2%}",
        f"Proporção de comentários neutros: {video_row.get('prop_neutro'):.2%}",
        f"Quantidade de comentários analisados: {video_row.get('qtd_comentarios_coletados')}",
    ]

    if feature_importance_summary:
        lines.append(f"Importância das features no modelo: {feature_importance_summary}")

    if similar_comments:
        lines.append("Comentários representativos deste vídeo:")
        lines.extend(f"- {c}" for c in similar_comments)

    lines.append("\nEm 2 a 3 frases, explique por que o modelo chegou a essa previsão de engajamento.")
    return "\n".join(lines)


def explain_prediction(
    video_row: pd.Series,
    feature_importance_summary: Optional[str] = None,
    retriever: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    model: str = config.ANTHROPIC_MODEL,
    client: Optional[anthropic.Anthropic] = None,
) -> str:
    """Gera, via Claude API, uma explicação em linguagem natural da previsão de engajamento de um vídeo."""
    similar_comments = None
    if retriever is not None:
        query = f"comentários sobre o vídeo {video_row.get('video_title', '')}"
        similar_comments = retrieve_top_k(retriever, query, top_k=top_k)

    prompt = build_prediction_prompt(video_row, feature_importance_summary, similar_comments)

    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


def explain_predictions(
    df_final: pd.DataFrame,
    model_results: Optional[Dict[str, Any]] = None,
    retriever: Optional[Dict[str, Any]] = None,
    top_k: int = 3,
    model: str = config.ANTHROPIC_MODEL,
) -> List[Dict[str, str]]:
    """Gera explicações em linguagem natural para as previsões de todos os vídeos em df_final."""
    reg_model = model_results.get('reg_model') if model_results else None
    feature_importance_summary = build_feature_importance_summary(reg_model)

    client = anthropic.Anthropic()
    explanations = []
    for _, row in df_final.iterrows():
        text = explain_prediction(row, feature_importance_summary, retriever, top_k, model, client)
        explanations.append({'video_id': row.get('video_id'), 'explanation': text})

    return explanations
