# Fluxo Completo do Pipeline

Documentação detalhada de `run_pipeline()` (`src/pipeline.py`), etapa por etapa: função chamada, entradas exatas, saídas exatas e efeitos colaterais. Para a visão geral em diagrama, ver [`architecture.md`](architecture.md).

Ponto de entrada: `main.py` → `run_pipeline() -> Tuple[df_comments, df_final, retriever]`.

---

## Etapa 1 — Coleta

**Módulo:** `src/youtube_collector.py`
**Função:** `get_video_comments_and_stats(api_key: str, video_ids: List[str], max_comments: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]`
**Chamada no pipeline:** `get_video_comments_and_stats(config.YOUTUBE_API_KEY, config.VIDEO_IDS, config.MAX_COMMENTS_PER_VIDEO)`

| | |
|---|---|
| **Entrada** | `api_key` (`config.YOUTUBE_API_KEY`, lido de env var) · `video_ids` (`config.VIDEO_IDS`, lista de strings) · `max_comments` (`config.MAX_COMMENTS_PER_VIDEO` = 100) |
| **Saída** | `df_comments`: colunas `video_id` (str), `text` (str) — um registro por comentário · `df_videos`: colunas `video_id`, `video_title`, `views`, `likes`, `total_comments_count` — um registro por vídeo |
| **Efeitos colaterais** | Chamadas HTTP à YouTube Data API v3. Falhas por vídeo (`HttpError`/`KeyError`/`ValueError`) são logadas e não interrompem o loop — o vídeo problemático é simplesmente omitido do resultado. |

**Checagem imediatamente após:** se `df_comments` estiver vazio, `run_pipeline()` levanta `ValueError` e interrompe todo o pipeline (chave de API inválida ou nenhum comentário coletado).

---

## Etapa 2 — Pré-processamento

**Módulo:** `src/text_preprocessing.py`
**Função:** `clean_comments(df: pd.DataFrame, text_column: str = "text", min_length: int = 3) -> pd.DataFrame`
**Chamada no pipeline:** `clean_comments(df_comments, min_length=config.MIN_CLEANED_TEXT_LENGTH)`

| | |
|---|---|
| **Entrada** | `df_comments` (etapa 1) · `min_length` (`config.MIN_CLEANED_TEXT_LENGTH` = 3) |
| **Saída** | `df_comments` + coluna `cleaned_text` (str); linhas cujo `cleaned_text` tem `len() <= min_length` são descartadas; índice resetado |
| **Efeitos colaterais** | Nenhum (função pura). Internamente aplica `clean_text()` célula a célula (remove URLs, `@menções`, `#hashtags`, caracteres especiais). |

---

## Etapa 3 — Análise de sentimentos

**Módulo:** `src/sentiment_analysis.py`
**Funções:** `load_sentiment_pipeline(model_name: str, device: int = -1) -> Any` e `analyze_sentiments(df, text_column, sentiment_pipeline, max_length=512) -> pd.DataFrame`
**Chamada no pipeline:** `load_sentiment_pipeline(config.SENTIMENT_MODEL_NAME)` seguido de `analyze_sentiments(df_comments, 'cleaned_text', sentiment_pipeline, config.MAX_TEXT_LENGTH_FOR_MODEL)`

| | |
|---|---|
| **Entrada (load)** | `model_name` (`config.SENTIMENT_MODEL_NAME`) · `device=-1` (força CPU) |
| **Saída (load)** | pipeline de `transformers` pronto para inferência |
| **Entrada (analyze)** | `df_comments` (com `cleaned_text`) · `text_column='cleaned_text'` · `max_length` (`config.MAX_TEXT_LENGTH_FOR_MODEL` = 512, trunca o texto antes de passar ao modelo) |
| **Saída (analyze)** | `df_comments` + colunas `label` (str) e `score` (float) |
| **Efeitos colaterais** | Baixa o modelo do Hugging Face na primeira execução. Barra de progresso (`tqdm`). Erros de inferência por comentário (`TypeError`/`ValueError`/`RuntimeError`/`KeyError`/`IndexError`) caem no fallback silencioso `{'label': 'NEUTRAL', 'score': 0.5}`. |

---

## Etapa 4 — Engenharia de atributos

**Módulo:** `src/feature_engineering.py`
**Funções:** `add_sentiment_flags`, `aggregate_by_video`, `build_final_dataset`

| Função | Entrada | Saída |
|---|---|---|
| `add_sentiment_flags(df)` | `df_comments` (com `label`) | `df_comments` + `is_pos`, `is_neg`, `is_neu` (int 0/1, via `label.str.upper()`) |
| `aggregate_by_video(df)` | `df_comments` (com as flags) | `agg_comments`: `video_id`, `prop_positivo`, `prop_negativo`, `prop_neutro` (médias), `qtd_comentarios_coletados` (contagem) |
| `build_final_dataset(df_videos, agg_comments)` | `df_videos` (etapa 1) + `agg_comments` | `df_final`: todas as colunas de `df_videos` e `agg_comments` (merge `inner` por `video_id`) + `engagement_rate` = `(likes + total_comments_count) / views` + `engagement_class` (1 se `engagement_rate` ≥ mediana, senão 0) |

`df_final` é o dataset final usado pela modelagem, visualização e comparação de modelos.

---

## Etapa 5 — Modelagem preditiva

**Módulo:** `src/modeling.py`
**Funções:** `train_and_evaluate(df_final, ...)` e `compare_models(df_final, ...)` — chamadas no pipeline sem overrides (usam todos os defaults).

| | |
|---|---|
| **Entrada (ambas)** | `df_final` · `feature_cols` (default `FEATURE_COLUMNS = ['prop_positivo', 'prop_negativo', 'prop_neutro', 'qtd_comentarios_coletados']`) · `test_size=0.2` · `random_state=42` |
| **Saída `train_and_evaluate`** | `model_results`: dict com `reg_model`, `clf_model` (estimadores treinados), `mae`, `rmse`, `r2_score`, `accuracy`, `f1_score`, `cv_r2_mean`, `cv_accuracy_mean`. **Dict vazio `{}`** se `len(df_final) < 2`. |
| **Saída `compare_models`** | `df_comparison`: DataFrame com colunas `tipo` (`regressao`/`classificacao`), `modelo` (`LinearRegression`/`LogisticRegression`/`RandomForest`/`XGBoost`), `mae`, `rmse`, `r2`, `accuracy`, `f1` (6 linhas — `NaN` nas métricas não aplicáveis ao tipo). **DataFrame vazio** (mesmas colunas) se `len(df_final) < 2`. |
| **Efeitos colaterais** | Imprime métricas de treino/validação cruzada no console (`train_and_evaluate`). `compare_models` não imprime nada. |

`train_and_evaluate` treina Random Forest com holdout (80/20) + validação cruzada K-Fold (`n_splits=5`, ajustado para `min(5, len(df_final))`). `compare_models` treina 3 modelos de regressão e 3 de classificação no mesmo holdout, sem validação cruzada.

---

## Etapa 6 — Exportação

**Módulo:** `src/export.py`

| Função | Entrada | Saída (arquivo) |
|---|---|---|
| `save_outputs(df_comments, df_final, output_dir)` | `df_comments`, `df_final`, `config.OUTPUT_DIR` (`"outputs"`) | `outputs/comentarios_processados.csv`, `outputs/dataset_final_videos.csv` |
| `save_metrics(metrics, output_dir)` — só se `model_results` não vazio | dict remapeado no pipeline: `{'mae', 'rmse', 'r2': model_results['r2_score'], 'f1': model_results['f1_score'], 'accuracy'}` · `outputs/metrics` | `outputs/metrics/metrics.csv` (1 linha) |
| `save_model_comparison(df_comparison, output_dir)` — só se `df_comparison` não vazio | `df_comparison` · `outputs/metrics` | `outputs/metrics/comparacao_modelos.csv` |

Todos gravados em `utf-8-sig` (compatível com Excel em PT-BR).

---

## Etapa 7 — Visualização

**Módulo:** `src/visualization.py`
**Função:** `generate_all_visualizations(df_comments, df_final, model_results, output_dir, df_comparison) -> None`
**Chamada no pipeline:** `generate_all_visualizations(df_comments, df_final, model_results, f"{config.OUTPUT_DIR}/figures", df_comparison)`

| Entrada | Origem |
|---|---|
| `df_comments` | etapa 3 (precisa da coluna `label`) |
| `df_final` | etapa 4 |
| `model_results` | etapa 5 (`train_and_evaluate`) |
| `df_comparison` | etapa 5 (`compare_models`) |

**Saída (arquivos em `outputs/figures/`):**

| Arquivo | Gerado por | Condição |
|---|---|---|
| `distribuicao_sentimentos.png` | `plot_sentiment_distribution` | sempre |
| `engajamento_por_video.png` | `plot_engagement_by_video` | sempre |
| `distribuicao_engagement_rate.png` | `plot_engagement_rate_distribution` | sempre |
| `sentimento_vs_engajamento.png` | `plot_sentiment_vs_engagement` | sempre |
| `matriz_correlacao.png` | `plot_correlation_matrix` | sempre |
| `importancia_features_regressao.png` | `plot_feature_importance` | se `model_results` não vazio e `reg_model` tiver `feature_importances_` |
| `importancia_features_classificacao.png` | `plot_feature_importance` | se `model_results` não vazio e `clf_model` tiver `feature_importances_` |
| `comparacao_modelos.png` | `plot_model_comparison` | se `df_comparison` não vazio |

---

## Etapa 8 — RAG (indexação)

**Módulo:** `src/rag/rag_pipeline.py`
**Função:** `run_rag_pipeline(df_comments: pd.DataFrame, text_column: str = 'cleaned_text') -> Dict[str, Any]`
**Chamada no pipeline:** `run_rag_pipeline(df_comments)`

| | |
|---|---|
| **Entrada** | `df_comments` (etapa 3, precisa da coluna `cleaned_text`) |
| **Saída** | `retriever`: dict `{'comments': List[str], 'model': SentenceTransformer, 'index': faiss.Index}` |
| **Efeitos colaterais** | Baixa o modelo de embeddings na primeira execução; imprime progresso. **Não afeta nem é afetado pelas etapas 5–7** — é uma ramificação independente que só depende de `df_comments`. |

Este `retriever` indexa **todos os comentários de todos os vídeos juntos** (sem filtro por `video_id`) — ver [`src/rag/README.md`](../src/rag/README.md) para detalhes e limitações de uso combinado com `rag/explainer.py`.

---

## Retorno final

```python
def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    ...
    return df_comments, df_final, retriever
```

`main.py` chama `run_pipeline()` sem capturar o retorno — os três valores ficam disponíveis apenas para uso interativo/programático (ex.: notebook, REPL, ou testes).
