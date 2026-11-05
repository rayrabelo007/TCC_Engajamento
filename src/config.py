# ==========================================
# CONFIGURAÇÃO: INSIRA SEUS DADOS AQUI!
# ==========================================
import os
from typing import List

YOUTUBE_API_KEY: str = "AIzaSyAvMWm8V7wDWVYkScpzbsKFMvGTaMJL8Go"

# Coloque aqui os IDs dos vídeos que você quer analisar para o seu TCC
VIDEO_IDS = [
    "HhesaQXLuRY",
    "5MgBikgcWnY",
    "aircAruvnKk",
    "2ePf9rue1Ao",
    "kngZEB_7NqQ",
    "COY9z4yP7g8",
    "ix9cRaBkVe0",
    "8mAitjgyCs0",
    "7S_tuE2EEhU",
    "2-Ol9irblqk",
    "8mei6uVttho",
    "rfscVS0vtbw",
    "PkZtCw7ugWg",
    "V6m9vS9mZow",
    "M9vT4zYmZxw",
    "Kx_vS9mZ1ab"
    "V-_O7nl0Ii0",
    "8nd_hIBq_QE",
    "kqtD5dpn9EE",
    "Y75v_Jm_dUA",
    "3JZ_D3ELwOQ",
    "kJQP7kiw5Fk",
    "5qap5aO4i9A",
    "9bZkp7q19f0",
    "JGwWNGJdvx8",
    "aircAruvnKk",
    "jGw8ccgT4s0",
    "bBC-nXj3Ng4",
    "rfscVS0vtbw",
    "_uQrJ0TkZlc",
    "7eh4d6sabA0",
    "i_LwzRVP7bg",
    "NWONeJKn6kc",
    "fuiJ-3CdU3Q",
    "cbe84W30GM8",
    "O5HX80yNOU8",
    "KD8zHq37dfA",
    "zOjov-2OZ0E",
    "8j0UDiN7my4",
    "W6NZfCO5SIk",
    "PkZtNgRSZU8",
    "7S_tz1z_5bA",
    "HXw9HwTfb7g",
    "R9W9_QzJvYg"
]

# Modelo ideal para múltiplos idiomas incluindo Português
SENTIMENT_MODEL_NAME: str = "pysentimiento/bertweet-pt-sentiment"

MAX_COMMENTS_PER_VIDEO: int = 100
MIN_CLEANED_TEXT_LENGTH: int = 3
MAX_TEXT_LENGTH_FOR_MODEL: int = 512

OUTPUT_DIR: str = "outputs"

# Usado para gerar explicações em linguagem natural das previsões do modelo (ANTHROPIC_API_KEY via variável de ambiente)
ANTHROPIC_MODEL: str = "claude-opus-4-8"
