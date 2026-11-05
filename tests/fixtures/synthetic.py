"""Gerador de dados sintéticos com relação causal conhecida entre sentimento e engajamento.

Usado para validar o pipeline (Tarefa 0 de TAREFAS.md) sem depender de dados reais,
modelo de sentimento real ou decisão de nicho. Nenhuma chamada de rede é feita aqui.
"""

import numpy as np
import pandas as pd


def make_synthetic_dataset(n_videos: int, effect_size: float = 0.4, seed: int = 42):
    """Gera (df_comments, df_videos) sintéticos com relação CONHECIDA.

    Modelo gerador:
        prop_positivo ~ Beta(2, 2)
        log_inscritos ~ Normal(11, 1.5)
        idade_dias    ~ Uniform(30, 90)
        ruido         ~ Normal(0, 0.01)

        engagement_rate = 0.02
                        + effect_size * 0.03 * prop_positivo   <- efeito do sentimento
                        + 0.004 * (log_inscritos - 11)         <- confundidor
                        - 0.0001 * idade_dias
                        + ruido

    Com effect_size=0.0, o sentimento NÃO tem efeito. Um pipeline honesto
    deve falhar em detectar efeito nesse caso (ver test_no_false_positive).
    """
    rng = np.random.default_rng(seed)

    prop_positivo = rng.beta(2, 2, n_videos)
    log_inscritos = rng.normal(11, 1.5, n_videos)
    idade_dias = rng.uniform(30, 90, n_videos)
    duracao_seg = rng.uniform(60, 1200, n_videos)
    ruido = rng.normal(0, 0.01, n_videos)

    engagement_rate = (
        0.02
        + (effect_size * 0.03 * prop_positivo)
        + (0.004 * (log_inscritos - 11))
        - (0.0001 * idade_dias)
        + ruido
    )
    engagement_rate = np.clip(engagement_rate, 0.0001, None)

    inscritos = np.exp(log_inscritos)
    views = np.round(inscritos * rng.uniform(0.05, 0.5, n_videos)).clip(min=50)
    likes = np.round(engagement_rate * views).astype(int)

    video_ids = [f"synthetic_{i:05d}" for i in range(n_videos)]

    df_videos = pd.DataFrame({
        "video_id": video_ids,
        "views": views.astype(int),
        "likes": likes,
        "log_inscritos": log_inscritos,
        "idade_dias": idade_dias,
        "duracao_seg": duracao_seg,
        "engagement_rate": engagement_rate,
        # incluída só para os testes desta tarefa validarem a fixture;
        # não é uma feature do pipeline final (seria vazamento, ver Tarefa 2).
        "prop_positivo": prop_positivo,
    })

    comments_per_video = rng.integers(5, 30, n_videos)
    rows = []
    for i, video_id in enumerate(video_ids):
        n_c = comments_per_video[i]
        p_pos = prop_positivo[i]
        p_rest = (1.0 - p_pos) / 2.0
        labels = rng.choice(
            ["positive", "negative", "neutral"],
            size=n_c,
            p=[p_pos, p_rest, p_rest],
        )
        for j, label in enumerate(labels):
            rows.append({
                "video_id": video_id,
                "text": f"comentário sintético {j}",
                "label": label,
            })

    df_comments = pd.DataFrame(rows, columns=["video_id", "text", "label"])
    return df_comments, df_videos


def make_leaky_dataset(n_videos: int, effect_size: float = 0.4, seed: int = 42):
    """Idêntico a make_synthetic_dataset, mas com uma feature que é função
    direta do alvo (`qtd_comentarios_coletados`), simulando o vazamento que
    a Tarefa 2 corrige (lá, a versão real é a contagem de comentários entrando
    tanto na fórmula do alvo quanto como feature). Serve para provar que o
    teste de vazamento pega o problema: uma regressão treinada com essa
    feature atinge R² absurdamente alto.
    """
    df_comments, df_videos = make_synthetic_dataset(n_videos, effect_size, seed)

    rng = np.random.default_rng(seed + 1)
    leak_noise = rng.normal(0, 1e-6, n_videos)
    df_videos["qtd_comentarios_coletados"] = (
        (df_videos["engagement_rate"] * 100_000 + leak_noise)
        .round()
        .astype(int)
        .clip(lower=0)
    )
    return df_comments, df_videos


def sample_size_status(n: int) -> dict:
    """Guarda de tamanho amostral.

    Preview do que a Tarefa 4 vai formalizar dentro de src/modeling.py
    (compare_models). Definida aqui só para que os testes desta tarefa
    possam validar a fixture de ponta a ponta antes de Tarefas 1-4 existirem.

        n < 8       -> RuntimeError, não treina.
        8 <= n < 50 -> confiavel=False.
        n >= 50     -> confiavel=True.
    """
    if n < 8:
        raise RuntimeError(f"Amostra insuficiente para treinar (n={n} < 8).")
    if n < 50:
        return {"confiavel": False, "n": n}
    return {"confiavel": True, "n": n}