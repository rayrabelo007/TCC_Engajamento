# ==========================================
# 7) VISUALIZAÇÃO (GRÁFICOS)
# ==========================================
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 (precisa vir depois de matplotlib.use)
from matplotlib.figure import Figure  # noqa: E402

from src.modeling import FEATURE_COLUMNS  # noqa: E402


def _save_and_close(fig: Figure, output_dir: str, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(f"{output_dir}/{filename}")
    plt.close(fig)


def plot_sentiment_distribution(df_comments: pd.DataFrame, output_dir: str) -> None:
    """Gera um gráfico de barras e pizza com a distribuição de sentimentos dos comentários."""
    counts = df_comments['label'].value_counts()

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_bar.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax_bar.set_title("Distribuição de Sentimentos (Barras)")
    ax_bar.set_xlabel("Sentimento")
    ax_bar.set_ylabel("Quantidade de Comentários")

    ax_pie.pie(counts.values, labels=counts.index.astype(str), autopct='%1.1f%%', startangle=90)
    ax_pie.set_title("Distribuição de Sentimentos (Pizza)")
    ax_pie.axis('equal')

    _save_and_close(fig, output_dir, "distribuicao_sentimentos.png")


def plot_engagement_by_video(df_final: pd.DataFrame, output_dir: str) -> None:
    """Gera um gráfico de barras com a taxa de engajamento por vídeo, ordenado decrescente."""
    df_sorted = df_final.sort_values('engagement_rate', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_sorted['video_title'].astype(str), df_sorted['engagement_rate'], color="#55A868")
    ax.set_title("Taxa de Engajamento por Vídeo")
    ax.set_xlabel("Vídeo")
    ax.set_ylabel("Taxa de Engajamento")
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    _save_and_close(fig, output_dir, "engajamento_por_video.png")


def plot_engagement_rate_distribution(
    df_final: pd.DataFrame, output_dir: str, filename: str = "distribuicao_engagement_rate.png"
) -> None:
    """Gera histograma e boxplot da distribuição da taxa de engajamento."""
    values = df_final['engagement_rate']
    bins = max(1, min(10, len(values)))

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_hist.hist(values, bins=bins, color="#DD8452", edgecolor="black")
    ax_hist.set_title("Distribuição da Taxa de Engajamento (Histograma)")
    ax_hist.set_xlabel("Taxa de Engajamento")
    ax_hist.set_ylabel("Frequência")

    ax_box.boxplot(values, vert=True)
    ax_box.set_title("Distribuição da Taxa de Engajamento (Boxplot)")
    ax_box.set_ylabel("Taxa de Engajamento")
    ax_box.set_xticklabels(["engagement_rate"])

    _save_and_close(fig, output_dir, filename)


def plot_sentiment_vs_engagement(df_final: pd.DataFrame, output_dir: str) -> None:
    """Gera um gráfico de dispersão de sentimento positivo vs taxa de engajamento, com linha de tendência."""
    x = df_final['prop_positivo']
    y = df_final['engagement_rate']

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, color="#C44E52")
    if len(df_final) >= 2:
        coeffs = np.polyfit(x, y, 1)
        trend_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(trend_x, np.polyval(coeffs, trend_x), color="#333333", linestyle="--")
    ax.set_title("Proporção de Sentimento Positivo vs Taxa de Engajamento")
    ax.set_xlabel("Proporção de Comentários Positivos")
    ax.set_ylabel("Taxa de Engajamento")
    _save_and_close(fig, output_dir, "sentimento_vs_engajamento.png")


def plot_correlation_matrix(
    df_final: pd.DataFrame, output_dir: str, filename: str = "matriz_correlacao.png"
) -> None:
    """Gera um heatmap com a matriz de correlação entre as colunas numéricas de df_final."""
    corr = df_final.select_dtypes(include=[np.number]).corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black', fontsize=7)
    ax.set_title("Matriz de Correlação")
    fig.colorbar(im, ax=ax)
    _save_and_close(fig, output_dir, filename)


def plot_feature_importance(
    model: Optional[Any], feature_cols: List[str], output_dir: str, filename: str, title: str
) -> None:
    """Gera um gráfico de barras horizontais com a importância das features de um modelo treinado."""
    if model is None or not hasattr(model, 'feature_importances_'):
        return

    importances = model.feature_importances_
    order = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(feature_cols)[order], importances[order], color="#8172B2")
    ax.set_title(title)
    ax.set_xlabel("Importância")
    _save_and_close(fig, output_dir, filename)


def plot_model_comparison(
    df_comparison: Optional[pd.DataFrame], output_dir: str, filename: str = "comparacao_modelos.png"
) -> None:
    """Gera gráficos comparando R² (regressão) e Acurácia/F1 (classificação) entre os modelos avaliados."""
    if df_comparison is None or df_comparison.empty:
        return

    df_reg = df_comparison[df_comparison['tipo'] == 'regressao']
    df_clf = df_comparison[df_comparison['tipo'] == 'classificacao']

    fig, (ax_reg, ax_clf) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_reg.bar(df_reg['modelo'], df_reg['r2'], color="#4C72B0")
    ax_reg.set_title("Comparação de Modelos - Regressão")
    ax_reg.set_xlabel("Modelo")
    ax_reg.set_ylabel("R² Score")
    ax_reg.tick_params(axis='x', rotation=30)

    x = np.arange(len(df_clf))
    width = 0.35
    ax_clf.bar(x - width / 2, df_clf['accuracy'], width, label='Acurácia', color="#55A868")
    ax_clf.bar(x + width / 2, df_clf['f1'], width, label='F1', color="#C44E52")
    ax_clf.set_xticks(x)
    ax_clf.set_xticklabels(df_clf['modelo'], rotation=30)
    ax_clf.set_title("Comparação de Modelos - Classificação")
    ax_clf.set_ylabel("Score")
    ax_clf.legend()

    _save_and_close(fig, output_dir, filename)


def generate_all_visualizations(
    df_comments: pd.DataFrame,
    df_final: pd.DataFrame,
    model_results: Optional[Dict[str, Any]] = None,
    output_dir: str = "outputs/figures",
    df_comparison: Optional[pd.DataFrame] = None,
) -> None:
    """Gera e salva todos os gráficos do pipeline (sentimento, engajamento, correlação, features, comparação)."""
    os.makedirs(output_dir, exist_ok=True)

    plot_sentiment_distribution(df_comments, output_dir)
    plot_engagement_by_video(df_final, output_dir)
    plot_engagement_rate_distribution(df_final, output_dir)
    plot_sentiment_vs_engagement(df_final, output_dir)
    plot_correlation_matrix(df_final, output_dir)
    if model_results:
        plot_feature_importance(
            model_results.get('reg_model'), FEATURE_COLUMNS, output_dir,
            "importancia_features_regressao.png", "Importância das Features - Regressão (Random Forest)"
        )
        plot_feature_importance(
            model_results.get('clf_model'), FEATURE_COLUMNS, output_dir,
            "importancia_features_classificacao.png", "Importância das Features - Classificação (Random Forest)"
        )
    plot_model_comparison(df_comparison, output_dir)

    print(f"[SUCESSO] Gráficos salvos em '{output_dir}'!")
