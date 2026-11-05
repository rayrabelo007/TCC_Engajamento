# ==========================================
# 4) ENGENHARIA DE ATRIBUTOS (AGREGAÇÃO)
# ==========================================
import pandas as pd


def add_sentiment_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas binárias is_pos/is_neg/is_neu a partir da coluna label."""
    df = df.copy()
    # Transforma as labels em colunas binárias para calcular proporção
    df['is_pos'] = (df['label'].str.upper() == 'POS').astype(int)
    df['is_neg'] = (df['label'].str.upper() == 'NEG').astype(int)
    df['is_neu'] = (df['label'].str.upper() == 'NEU').astype(int)
    return df


def aggregate_by_video(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega os comentários por vídeo, calculando proporções de sentimento e quantidade coletada."""
    return df.groupby('video_id').agg(
        prop_positivo=('is_pos', 'mean'),
        prop_negativo=('is_neg', 'mean'),
        prop_neutro=('is_neu', 'mean'),
        qtd_comentarios_coletados=('text', 'count')
    ).reset_index()


def build_final_dataset(df_videos: pd.DataFrame, agg_comments: pd.DataFrame) -> pd.DataFrame:
    """Junta as métricas do vídeo com a agregação de sentimento e calcula engagement_rate/engagement_class."""
    # Junta com as métricas do vídeo
    df_final = pd.merge(df_videos, agg_comments, on='video_id', how='inner')

    # CÁLCULO DA MÉTRICA ALVO: TAXA DE ENGAJAMENTO
    df_final['engagement_rate'] = (df_final['likes'] + df_final['total_comments_count']) / df_final['views']
    
    # CLASSE ALVO PARA CLASSIFICAÇÃO:
    # Utiliza-se a mediana da amostra como ponto de corte (threshold) devido à ausência 
    # de um padrão universal de mercado para classificação de alto/baixo engajamento em vídeos digitais.
    median_engagement = df_final['engagement_rate'].median()
    df_final['engagement_class'] = (df_final['engagement_rate'] >= median_engagement).astype(int)
    
    return df_final