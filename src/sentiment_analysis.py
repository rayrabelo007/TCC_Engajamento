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
            raw_label = str(res['label']).upper()
            
            # Normalização de labels para garantir compatibilidade com a engenharia de atributos:
            # Transforma variações comuns do Hugging Face em POS, NEG ou NEU
            if 'POS' in raw_label or raw_label == 'LABEL_2':
                label = 'POS'
            elif 'NEG' in raw_label or raw_label == 'LABEL_0':
                label = 'NEG'
            else:
                label = 'NEU'
                
            sentiments.append({'label': label, 'score': res['score']})
            
        except (TypeError, ValueError, RuntimeError, KeyError, IndexError):
            # Fallback seguro em caso de falha na inferência
            sentiments.append({'label': 'NEU', 'score': 0.5})

    df_sent = pd.DataFrame(sentiments)
    return pd.concat([df, df_sent], axis=1)