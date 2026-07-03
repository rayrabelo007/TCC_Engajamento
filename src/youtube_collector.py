# ==========================================
# 1) FUNÇÃO PARA COLETAR COMENTÁRIOS
# ==========================================
from typing import List, Tuple

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def get_video_comments_and_stats(
    api_key: str, video_ids: List[str], max_comments: int = 100
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coleta comentários e estatísticas (views, likes, título) de uma lista de vídeos do YouTube."""
    youtube = build('youtube', 'v3', developerKey=api_key)
    all_comments = []
    video_stats_dict = {}

    print("=> Iniciando coleta de dados do YouTube...")
    for v_id in video_ids:
        try:
            # Coleta Estatísticas do Vídeo (Views, Likes)
            v_request = youtube.videos().list(part="statistics,snippet", id=v_id)
            v_response = v_request.execute()

            if not v_response['items']:
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

            # Coleta os Comentários (Até 100 por vídeo para teste)
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
            # HttpError: falha na chamada à API do YouTube (cota, permissão, ID inválido, comentários desativados)
            # KeyError: resposta da API sem os campos esperados
            # ValueError: campo numérico (views/likes/comentários) em formato inesperado
            print(f"Erro ao processar o vídeo {v_id}: {e}")

    df_comments = pd.DataFrame(all_comments)
    df_videos = pd.DataFrame.from_dict(video_stats_dict, orient='index').reset_index()
    df_videos = df_videos.rename(columns={'index': 'video_id'})
    return df_comments, df_videos
