# Módulo RAG (`src/rag/`)

Camada de Retrieval-Augmented Generation construída sobre os comentários coletados pelo pipeline principal (`src/pipeline.py`). É independente da parte de Machine Learning (`src/modeling.py`): não altera features, modelos ou métricas — apenas consome `df_comments`/`df_final` já prontos e, opcionalmente, os resultados do treino (`model_results`) para gerar explicações em linguagem natural.

## Arquitetura

```
comentários (df_comments['cleaned_text'])
        │
        ▼
┌─────────────────┐
│ embeddings.py    │  SentenceTransformer (multilíngue) → vetores densos
└────────┬─────────┘
         ▼
┌─────────────────┐
│ vector_store.py  │  índice FAISS (IndexFlatL2) → build / search / save / load
└────────┬─────────┘
         ▼
┌─────────────────┐
│ retriever.py     │  junta embeddings + vector_store → busca top-k por similaridade
└────────┬─────────┘
         ▼
┌─────────────────┐
│ rag_pipeline.py  │  orquestra a indexação (etapa 8 do pipeline principal)
└────────┬─────────┘
         ▼
┌─────────────────┐
│ explainer.py     │  usa o retriever + Claude API para explicar as previsões do modelo
└──────────────────┘
```

Fluxo de dependências entre os módulos (cada um importa só o anterior):

`embeddings.py` → `vector_store.py` → `retriever.py` → `rag_pipeline.py` / `explainer.py`

## Módulos

### `embeddings.py`
- `load_embedding_model(model_name=DEFAULT_MODEL_NAME)` — carrega um `SentenceTransformer`. Padrão: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilíngue, bom suporte a PT-BR, coerente com o modelo de sentimento usado no restante do projeto).
- `generate_embeddings(texts, model)` — retorna os vetores (`numpy.ndarray`, shape `(n, 384)`).

### `vector_store.py`
Wrapper fino sobre o FAISS:
- `build_index(embeddings)` — cria um `faiss.IndexFlatL2` e adiciona os vetores.
- `search(index, query_embedding, top_k=5)` — retorna `(distances, indices)` dos vizinhos mais próximos.
- `save_index(index, path)` / `load_index(path)` — persistência em disco.

### `retriever.py`
Combina os dois módulos acima em uma unidade reutilizável:
- `build_retriever(comments, model=None)` — gera embeddings de uma lista de comentários e monta o índice. Retorna um dict `{'comments', 'model', 'index'}`.
- `retrieve_top_k(retriever, query, top_k=5)` — embeda a query e retorna os `top_k` comentários mais similares (filtra o padding `-1` que o FAISS retorna quando `top_k` excede o total de comentários indexados).

### `rag_pipeline.py`
Orquestração de alto nível, análoga ao `src/pipeline.py` do restante do projeto:
- `run_rag_pipeline(df_comments, text_column='cleaned_text')` — indexa a coluna de texto limpo do `df_comments` e retorna o `retriever` pronto para uso. É a etapa 8 do `run_pipeline()` principal.

### `explainer.py`
Camada de geração (a parte "G" do RAG), via [Claude API](https://platform.claude.com) (`anthropic` SDK):
- `build_feature_importance_summary(reg_model)` — resume a importância das features do `RandomForestRegressor` treinado.
- `build_prediction_prompt(video_row, feature_importance_summary, similar_comments)` — monta o prompt (métricas do vídeo + importância das features + comentários recuperados via `retriever.retrieve_top_k`).
- `explain_prediction(video_row, ..., retriever=None, model=config.ANTHROPIC_MODEL)` — chama `client.messages.create` e retorna a explicação em texto.
- `explain_predictions(df_final, model_results, retriever=None)` — roda a explicação para todos os vídeos de `df_final`, reaproveitando um único client HTTP.

Modelo padrão: `claude-opus-4-8` (configurável em `src/config.py`). A chave é lida de `ANTHROPIC_API_KEY` no ambiente — não é hardcoded em nenhum arquivo.

## Como usar

```python
from src.rag.rag_pipeline import run_rag_pipeline
from src.rag.retriever import retrieve_top_k
from src.rag.explainer import explain_predictions

# 1) Indexar os comentários já coletados/limpos
retriever = run_rag_pipeline(df_comments)

# 2) Buscar comentários similares a uma consulta livre
resultados = retrieve_top_k(retriever, "reclamações sobre entrega", top_k=5)

# 3) Gerar explicações em linguagem natural das previsões do modelo
explicacoes = explain_predictions(df_final, model_results, retriever=retriever)
```

## Dependências

| Pacote | Papel |
|---|---|
| `sentence-transformers` | geração de embeddings |
| `faiss-cpu` | índice vetorial e busca por similaridade |
| `anthropic` | geração das explicações (LLM) |

Todas listadas em `requirements.txt` (raiz do projeto).

## Estado atual / limitações conhecidas

- `run_rag_pipeline` é chamado automaticamente pelo pipeline principal (etapa 8), mas o `retriever` resultante ainda não é usado em nenhuma etapa downstream por padrão — fica disponível no retorno de `run_pipeline()` para uso interativo/futuro.
- `explain_predictions` não é chamado pelo pipeline principal; precisa ser invocado manualmente (custo de API por chamada).
- O índice FAISS é reconstruído do zero a cada execução (não há cache entre rodadas), já que `save_index`/`load_index` existem mas não são usados automaticamente.
- **Sem escopo por vídeo**: `build_retriever` indexa os comentários de **todos os vídeos juntos** — `retriever['comments']` é uma lista plana de texto, sem `video_id` associado a cada entrada. `explain_prediction` busca nesse índice único usando só o título do vídeo como proxy de consulta (`f"comentários sobre o vídeo {titulo}"`); como não há filtro por `video_id`, os "comentários representativos" retornados para um vídeo podem, na prática, pertencer a outro vídeo do dataset. Para corrigir: guardar `video_id` junto de cada comentário no `retriever` e filtrar `retrieve_top_k` pelo vídeo da previsão sendo explicada.

## Documentação relacionada

- [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — arquitetura completa do projeto (árvore, diagrama, todas as etapas).
- [`../../docs/dependencies.md`](../../docs/dependencies.md) — grafo de imports entre todos os módulos, incluindo `src/rag/`.
- [`../../docs/ml.md`](../../docs/ml.md) — componentes de Machine Learning que `explainer.py` consome (`FEATURE_COLUMNS`, `reg_model.feature_importances_`).
