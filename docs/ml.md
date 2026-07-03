# Componentes de Machine Learning

Documentação de `src/modeling.py` (treino/avaliação) e da parte de `src/export.py` que persiste os resultados de ML. Para os dados de entrada (como `df_final` é construído), ver [`pipeline.md`](pipeline.md) — Etapa 4.

## Problema e alvos

Duas tarefas de ML sobre o mesmo dataset (`df_final`, uma linha por vídeo):

| Tarefa | Alvo (coluna) | Definição |
|---|---|---|
| Regressão | `engagement_rate` | `(likes + total_comments_count) / views` |
| Classificação binária | `engagement_class` | `1` se `engagement_rate` ≥ mediana do dataset, senão `0` |

**Features de entrada** (`FEATURE_COLUMNS`, constante em `modeling.py`, reutilizada por `visualization.py` e `rag/explainer.py`):

```python
FEATURE_COLUMNS = ['prop_positivo', 'prop_negativo', 'prop_neutro', 'qtd_comentarios_coletados']
```

As três primeiras são proporções de sentimento (calculadas em `feature_engineering.aggregate_by_video`); a quarta é a contagem de comentários coletados por vídeo.

## Funções auxiliares compartilhadas

Usadas tanto por `train_and_evaluate` quanto por `compare_models` (extraídas para eliminar duplicação):

| Função | Assinatura | O que faz |
|---|---|---|
| `prepare_features_and_targets` | `(df_final, feature_cols=FEATURE_COLUMNS) -> (X, y_reg, y_clf)` | Fatia `df_final` nas colunas de features e nos dois alvos. |
| `compute_regression_metrics` | `(y_true, y_pred) -> {'mae', 'rmse', 'r2'}` | `mean_absolute_error`, `sqrt(mean_squared_error)`, `r2_score`. |
| `compute_classification_metrics` | `(y_true, y_pred) -> {'accuracy', 'f1'}` | `accuracy_score`, `f1_score(zero_division=0)`. |

## Treino e avaliação — `train_and_evaluate`

```python
train_and_evaluate(df_final, feature_cols=FEATURE_COLUMNS, test_size=0.2, random_state=42, n_splits=5) -> Dict[str, Any]
```

Modelo único por tarefa: **`RandomForestRegressor`** e **`RandomForestClassifier`** (`sklearn.ensemble`, `random_state=42`).

**Guarda de tamanho mínimo:** só treina se `len(df_final) >= 2`; caso contrário imprime um aviso e retorna `{}` (dict vazio).

Duas avaliações independentes são feitas sobre os mesmos dados:

1. **Holdout 80/20** (`train_test_split`, `test_size=0.2`): treina em `X_train`/`y_train`, avalia em `X_test`/`y_test` — a métrica reflete generalização para dados não vistos.
2. **Validação cruzada K-Fold** (`KFold(n_splits=min(5, len(df_final)), shuffle=True, random_state=42)`, via `cross_val_score`): treina/avalia `k` vezes sobre **todo** `X`/`y` (não usa o split do holdout), dando uma estimativa mais robusta com poucos dados. Dois novos modelos `RandomForest*` são instanciados só para isso (não são os mesmos objetos retornados em `reg_model`/`clf_model`).

**Retorno** (`model_results`, dict):

| Chave | Origem |
|---|---|
| `reg_model`, `clf_model` | Os dois estimadores treinados no holdout (podem ser reutilizados, ex.: `reg_model.feature_importances_`) |
| `mae`, `rmse`, `r2_score` | Métricas de regressão no holdout |
| `accuracy`, `f1_score` | Métricas de classificação no holdout |
| `cv_r2_mean`, `cv_accuracy_mean` | Médias da validação cruzada K-Fold |

> Note a assimetria de nomes: aqui as chaves são `r2_score`/`f1_score`; em `compare_models` (abaixo) as colunas equivalentes se chamam `r2`/`f1`. `pipeline.py` faz esse remapeamento manualmente ao chamar `save_metrics`.

## Comparação entre modelos — `compare_models`

```python
compare_models(df_final, feature_cols=FEATURE_COLUMNS, test_size=0.2, random_state=42) -> pd.DataFrame
```

Treina **6 modelos** no mesmo holdout 80/20 (sem validação cruzada):

| Tipo | Modelos |
|---|---|
| Regressão | `LinearRegression`, `RandomForestRegressor`, `XGBRegressor` |
| Classificação | `LogisticRegression(max_iter=1000)`, `RandomForestClassifier`, `XGBClassifier(eval_metric='logloss')` |

**Retorno:** `DataFrame` com colunas `tipo` (`regressao`/`classificacao`), `modelo`, `mae`, `rmse`, `r2`, `accuracy`, `f1` — 6 linhas (uma por modelo), com `NaN` nas métricas que não se aplicam ao tipo (ex.: uma linha `regressao` tem `accuracy`/`f1` = `NaN`).

**Guarda de tamanho mínimo:** `len(df_final) < 2` retorna um `DataFrame` vazio (mesmas colunas, zero linhas) — igual em espírito ao `{}` de `train_and_evaluate`, mas **essa guarda não é suficiente**: veja a limitação abaixo.

## ⚠️ Limitação conhecida: `compare_models` pode falhar com poucas amostras

Com `len(df_final) == 2` (a configuração padrão do repositório, que vem com 2 `VIDEO_IDS` de exemplo), o holdout 80/20 produz **1 amostra de treino**. Se essa única amostra de treino tiver só uma classe de `engagement_class`, `LogisticRegression.fit()` lança:

```
ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class
```

Isso não é capturado — propaga e derruba `run_pipeline()` inteiro. `RandomForestRegressor`/`RandomForestClassifier` não têm esse problema (toleram treino com uma classe só), então `train_and_evaluate` roda normalmente nesse mesmo cenário; só `compare_models` quebra. Correção sugerida: capturar `ValueError` por modelo (marcando a métrica como `NaN`, como já é feito na guarda de `len(df_final) < 2`) em vez de deixar propagar.

## Exportação dos resultados de ML

**Módulo:** `src/export.py` — chamado por `pipeline.py` logo após `train_and_evaluate`/`compare_models` (Etapa 6).

| Função | Entrada | Saída |
|---|---|---|
| `save_metrics(metrics, output_dir="outputs/metrics")` | dict remapeado no pipeline: `{'mae', 'rmse', 'r2': model_results['r2_score'], 'f1': model_results['f1_score'], 'accuracy'}` | `outputs/metrics/metrics.csv` (1 linha) — só chamado se `model_results` não for `{}` |
| `save_model_comparison(df_comparison, output_dir="outputs/metrics")` | `df_comparison` (retorno de `compare_models`) | `outputs/metrics/comparacao_modelos.csv` — só chamado se `df_comparison` não estiver vazio |

Ambos gravam em `utf-8-sig` (abre corretamente em Excel PT-BR) e criam o diretório se não existir (`os.makedirs(..., exist_ok=True)`).

## Consumo downstream dos resultados de ML

- **`visualization.py`** usa `model_results['reg_model']`/`['clf_model']` (via `.feature_importances_`) para os gráficos de importância de features, e `df_comparison` para o gráfico de comparação — ver [`pipeline.md`](pipeline.md) Etapa 7.
- **`rag/explainer.py`** usa `model_results['reg_model'].feature_importances_` (via `build_feature_importance_summary`) para incluir a importância das features no prompt de explicação por LLM — ver [`../src/rag/README.md`](../src/rag/README.md).
