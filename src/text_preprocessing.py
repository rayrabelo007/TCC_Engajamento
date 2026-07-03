# ==========================================
# 2) PRÉ-PROCESSAMENTO (LIMPEZA DO TEXTO)
# ==========================================
import re
from typing import Any

import pandas as pd


def clean_text(text: Any) -> str:
    """Remove URLs, menções, hashtags e caracteres especiais de um texto."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove @mencoes e #hashtags
    text = re.sub(r"[^\w\s,.:!?]", "", text, flags=re.UNICODE)  # Remove Emojis pesados
    text = re.sub(r"\s+", " ", text).strip()  # Remove espaços extras
    return text


def clean_comments(df: pd.DataFrame, text_column: str = "text", min_length: int = 3) -> pd.DataFrame:
    """Aplica clean_text à coluna de texto e descarta comentários muito curtos."""
    df = df.copy()
    df['cleaned_text'] = df[text_column].apply(clean_text)
    df = df[df['cleaned_text'].str.len() > min_length].reset_index(drop=True)
    return df
