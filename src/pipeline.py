# ==============================================================================
# PIPELINE COMPLETO - PREDIÇÃO DE ENGAJAMENTO
# ==============================================================================
from typing import Any, Dict, Tuple

import pandas as pd

from src import config
from src.youtube_collector import get_video_comments_and_stats
from src.text_preprocessing import clean_comments
from src.sentiment_analysis import load_sentiment_pipeline, analyze_sentiments
from src.feature_engineering import add_sentiment_flags, aggregate_by_video, build_final_dataset
from src.modeling import train_and_evaluate, compare_models
from src.export import save_outputs, save_metrics, save_model_comparison
from src.visualization import generate_all_visualizations
from src.rag.rag_pipeline import run_rag_pipeline


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Executa o pipeline completo: coleta, sentimento, features, ML, exportação, visualização e RAG."""
    # 1) Coleta
    df_comments, df_videos = get_video_comments_and_stats(
        config.YOUTUBE_API_KEY, config.VIDEO_IDS, config.MAX_COMMENTS_PER_VIDEO
    )

    if df_comments.empty:
        raise ValueError("Nenhum comentário foi coletado. Verifique sua Chave de API e os IDs dos vídeos na Célula 3!")

    # 2) Pré-processamento
    df_comments = clean_comments(df_comments, min_length=config.MIN_CLEANED_TEXT_LENGTH)

    # 3) Análise de sentimentos
    sentiment_pipeline = load_sentiment_pipeline(config.SENTIMENT_MODEL_NAME)
    df_comments = analyze_sentiments(
        df_comments, 'cleaned_text', sentiment_pipeline, config.MAX_TEXT_LENGTH_FOR_MODEL
    )

    # 4) Engenharia de atributos
    print("=> Agregando dados por nível de Vídeo...")
    df_comments = add_sentiment_flags(df_comments)
    agg_comments = aggregate_by_video(df_comments)
    df_final = build_final_dataset(df_videos, agg_comments)

    # 5) Modelagem preditiva
    model_results = train_and_evaluate(df_final)
    df_comparison = compare_models(df_final)

    # 6) Exportação
    save_outputs(df_comments, df_final, config.OUTPUT_DIR)
    if model_results:
        save_metrics({
            'mae': model_results['mae'],
            'rmse': model_results['rmse'],
            'r2': model_results['r2_score'],
            'f1': model_results['f1_score'],
            'accuracy': model_results['accuracy'],
        }, f"{config.OUTPUT_DIR}/metrics")
    if not df_comparison.empty:
        save_model_comparison(df_comparison, f"{config.OUTPUT_DIR}/metrics")

    # 7) Visualização
    generate_all_visualizations(
        df_comments, df_final, model_results, f"{config.OUTPUT_DIR}/figures", df_comparison
    )

    # 8) RAG (indexação dos comentários para recuperação por similaridade; não afeta o ML)
    retriever = run_rag_pipeline(df_comments)

    return df_comments, df_final, retriever
