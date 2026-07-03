# ==========================================
# CONFIGURAÇÃO: INSIRA SEUS DADOS AQUI!
# ==========================================
import os
from typing import List

YOUTUBE_API_KEY: str = os.environ.get("YOUTUBE_API_KEY", "")

# Coloque aqui os IDs dos vídeos que você quer analisar para o seu TCC
VIDEO_IDS: List[str] = [
    "HhesaQXLuRY",
    "5MgBikgcWnY",
    "aircAruvnKk",
    "2ePf9rue1Ao",
]

# Modelo ideal para múltiplos idiomas incluindo Português
SENTIMENT_MODEL_NAME: str = "pysentimiento/bertweet-pt-sentiment"

MAX_COMMENTS_PER_VIDEO: int = 100
MIN_CLEANED_TEXT_LENGTH: int = 3
MAX_TEXT_LENGTH_FOR_MODEL: int = 512

OUTPUT_DIR: str = "outputs"

# Usado para gerar explicações em linguagem natural das previsões do modelo (ANTHROPIC_API_KEY via variável de ambiente)
ANTHROPIC_MODEL: str = "claude-opus-4-8"
