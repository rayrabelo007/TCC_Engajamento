# ==========================================
# 3) ANÁLISE DE SENTIMENTOS (TRANSFORMERS)
# ==========================================
from typing import Any

import pandas as pd
from tqdm import tqdm
from transformers import pipeline


def load_sentiment_pipeline(model_name: str, device: int = -1) -> Any:
    """Carrega o pipeline de análise de sentimento do Hugging Face Transformers."""
    print("=> Carregando Modelo de Inteligência Artificial para Sentimentos...")
    return pipeline("sentiment-analysis", model=model_name, device=device)


def analyze_sentiments(
    df: pd.DataFrame, text_column: str, sentiment_pipeline: Any, max_length: int = 512
) -> pd.DataFrame:
    """Aplica o pipeline de sentimento a cada comentário e concatena as colunas label/score ao DataFrame."""
    print("=> Analisando sentimentos dos comentários (Isso pode demorar um pouco)...")
    sentiments = []
    for txt in tqdm(df[text_column]):
        try:
            res = sentiment_pipeline(txt[:max_length])[0]  # Limita o texto para o BERT
            sentiments.append({'label': res['label'], 'score': res['score']})
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
            # TypeError: texto não é string (ex.: NaN); ValueError/RuntimeError: falha de inferência do modelo;
            # KeyError/IndexError: saída do pipeline em formato inesperado
            sentiments.append({'label': 'NEUTRAL', 'score': 0.5})

    df_sent = pd.DataFrame(sentiments)
    return pd.concat([df, df_sent], axis=1)
