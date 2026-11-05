# ==========================================
# 1) FUNÇÃO PARA COLETAR COMENTÁRIOS
# ==========================================
import os
from typing import List, Tuple

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src import config


def get_video_comments_and_stats(*args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coleta comentários e estatísticas de vídeos do YouTube de forma flexível (aceita 2 ou 3 argumentos)."""
    api_key = None
    video_ids = None
    max_comments = config.MAX_COMMENTS_PER_VIDEO

    # Mapeia argumentos posicionais passados por pipeline.py
    if len(args) >= 2:
        if isinstance(args[0], str) and isinstance(args[1], list):
            api_key = args[0]
            video_ids = args[1]
            if len(args) >= 3:
                max_comments = args[2]
        elif isinstance(args[0], list):
            video_ids = args[0]
            if len(args) >= 2:
                max_comments = args[1]
    elif len(args) == 1 and isinstance(args[0], list):
        video_ids = args[0]

    # Mapeia argumentos nomeados (kwargs) se houver
    if 'api_key' in kwargs:
        api_key = kwargs['api_key']
    if 'video_ids' in kwargs:
        video_ids = kwargs['video_ids']
    if 'max_comments' in kwargs:
        max_comments = kwargs['max_comments']

    # Fallbacks padrão
    api_key = api_key or config.YOUTUBE_API_KEY or os.environ.get("YOUTUBE_API_KEY", "")
    video_ids = video_ids or config.VIDEO_IDS

    if not api_key:
        raise ValueError(
            "A chave da API do YouTube (YOUTUBE_API_KEY) não foi encontrada! "
            "Defina-a no arquivo config.py ou exporte a variável de ambiente YOUTUBE_API_KEY."
        )

    youtube = build('youtube', 'v3', developerKey=api_key)
    all_comments = []
    video_stats_dict = {}

    print("=> Iniciando coleta de dados do YouTube...")
    for v_id in video_ids:
        try:
            # Coleta Estatísticas do Vídeo (Views, Likes)
            v_request = youtube.videos().list(part="statistics,snippet", id=v_id)
            v_response = v_request.execute()

            if not v_response.get('items'):
                continue

            stats = v_response['items'][0]['statistics']
            title = v_response['items'][0]['snippet']['title']

            views = int(stats.get('viewCount', 0))
            likes = int(stats.get('likeCount', 0))
            comment_count = int(stats.get('commentCount', 0))

            video_stats_dict[v_id] = {
                'video_title': title,
                'views': views,
                'likes': likes,
                'total_comments_count': comment_count
            }

            # Coleta os Comentários
            c_request = youtube.commentThreads().list(
                part="snippet", videoId=v_id, maxResults=max_comments, textFormat="plainText"
            )
            c_response = c_request.execute()

            for item in c_response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                all_comments.append({
                    'video_id': v_id,
                    'text': comment
                })
        except (HttpError, KeyError, ValueError) as e:
            print(f"Erro ao processar o vídeo {v_id}: {e}")

    df_comments = pd.DataFrame(all_comments)
    df_videos = pd.DataFrame.from_dict(video_stats_dict, orient='index').reset_index()
    df_videos = df_videos.rename(columns={'index': 'video_id'})
    return df_comments, df_videos