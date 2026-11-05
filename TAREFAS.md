# TAREFAS.md — Plano de correção do pipeline

> **Para o Claude Code.** Este documento é a fonte de verdade sobre o que fazer e em que ordem.
> Leia junto com `CLAUDE.md` (regras não negociáveis) e `PROTOCOLO_AMOSTRAGEM.md` (decisões metodológicas).
> Se qualquer instrução aqui conflitar com `CLAUDE.md`, **pare e pergunte**.

---

## Estado do projeto — leia antes de tudo

**Ainda não há dados coletados.** A seleção dos vídeos está em andamento e depende de uma análise que ainda vai ser feita, com critérios que precisam de aprovação da orientadora.

Consequências operacionais, todas obrigatórias:

1. **Não chame a YouTube Data API.** `VIDEO_IDS` / `data/videos_selecionados.csv` permanecem vazios. Nenhuma tarefa abaixo requer coleta real.
2. **Toda validação é feita contra dados sintéticos** (Tarefa 0). Isso não é um paliativo: com dados sintéticos de relação conhecida, dá para verificar que o pipeline *recupera* o efeito que existe e *não inventa* efeito onde não existe. Nenhum dado real permite esse teste.
3. **Nada de números no TCC vindos de dados sintéticos.** Eles servem só para testar código. Marque isso claramente nos outputs.

## Regras de execução

- **Uma tarefa por sessão.** `/clear` entre elas. Não encadeie.
- **`/plan` antes de qualquer tarefa que toque mais de um arquivo.** Mostre o plano; espere aprovação.
- Ao final de cada tarefa: `python -m pytest -v` deve passar, depois `/diff`, depois commit com a mensagem indicada.
- **Nunca invente nomes de modelos do Hugging Face.** Se não tiver certeza de que o repositório existe, pare e pergunte.
- Se uma tarefa parecer exigir dados reais, você entendeu errado. Releia e pergunte.

---

## Tarefa 0 — Fixture sintética e testes de regressão

**Objetivo:** criar um gerador de dados falsos com relação causal conhecida, para que todas as tarefas seguintes possam ser verificadas sem API.

**Implementar** `tests/fixtures/synthetic.py`:

```python
def make_synthetic_dataset(n_videos: int, effect_size: float = 0.4, seed: int = 42):
    """
    Gera (df_comments, df_videos) sintéticos com relação CONHECIDA.

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
    deve falhar em detectar efeito nesse caso.
    """
```

**Implementar também** `make_leaky_dataset(...)`: idêntico, mas com uma feature que é função direta do alvo. Serve para provar que o teste de vazamento pega o problema.

**Testes obrigatórios** (`tests/test_synthetic.py`):

- `test_recovers_effect`: com `n_videos=500, effect_size=0.4`, o pipeline detecta correlação positiva significativa entre `prop_positivo` e `engagement_rate`.
- `test_no_false_positive`: com `n_videos=500, effect_size=0.0`, o pipeline **não** rejeita H₀ (p > 0.05). Este é o teste mais importante do repositório.
- `test_leakage_detected`: com `make_leaky_dataset`, o R² fica absurdamente alto (> 0.95) — demonstrando o sintoma que se quer evitar.
- `test_small_n_warns`: com `n_videos=10`, o pipeline emite aviso e se recusa a reportar métricas como confiáveis.

**Critério de aceite:** os quatro testes passam. `pytest` roda em < 60 s.

**Commit:** `test: fixture sintética com efeito conhecido e testes de regressão`

---

## Tarefa 1 — Corrigir a análise de sentimento

**Contexto:** `config.py` aponta hoje para `finiteautomata/bertwithsmiles-portuguese-tweets`, que **não existe**. Foi alucinado. O notebook original da orientadora usa um modelo real.

**Fazer:**

1. Em `config.py`, definir dois modelos e permitir alternar:
   - `SENTIMENT_MODEL_MULTILINGUAL = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"` — rótulos `positive` / `neutral` / `negative`. É o do notebook base.
   - `SENTIMENT_MODEL_PT = "pysentimiento/bertweet-pt-sentiment"` — rótulos `POS` / `NEG` / `NEU`. Baseado em BERTabaporu.
   - Manter os dois é requisito do Boneco §6.4 (comparar baseline e modelo avançado).

2. Escrever `normalize_sentiment_label(label: str) -> str` que mapeie **ambos** os conjuntos de rótulos para `positive` / `negative` / `neutral`. Case-insensitive. Deve tratar `POS`, `positive`, `LABEL_2`, etc.

3. Corrigir o fallback de erro de inferência: hoje devolve `{'label': 'NEUTRAL'}`, string que não existe em nenhum dos dois modelos. Trocar por `neutral` **e** contabilizar as falhas num contador logado ao final. Fallback silencioso que infla a classe neutra é inaceitável.

4. Em `text_preprocessing.py::clean_text`, **remover a remoção de caracteres especiais.** Emojis são sinal de sentimento em comentários de YouTube. Manter apenas: URLs, `@menções`, `#hashtags`, quebras de linha, espaços múltiplos. Exatamente como no notebook base.

5. Elevar `MIN_TEXT_LEN` de 3 para 5 (valor do notebook base).

**Testes** (`tests/test_sentiment.py`): 5 comentários em PT-BR com rótulo esperado; teste de que emojis sobrevivem ao `clean_text`; teste do mapeamento de rótulos para os dois vocabulários.

**Não fazer:** não baixe os modelos nos testes. Use `unittest.mock` para o `pipeline` do transformers.

**Commit:** `fix: modelo de sentimento inexistente, mapeamento de rótulos e preservação de emojis`

---

## Tarefa 2 — Eliminar o vazamento de alvo

**Contexto:** hoje o alvo é `(likes + total_comments_count) / views` e `qtd_comentarios_coletados` está entre as features. A feature integra o alvo. Além disso, a mediana que define `engagement_class` é calculada sobre o dataset inteiro, incluindo o teste.

**Fazer:**

1. Em `config.py`, criar `TARGET_FORMULA` com dois valores possíveis:
   - `"likes_only"` → `engagement_rate = likes / views` *(padrão)*
   - `"boneco"` → `engagement_rate = (likes + total_comments_count) / views`

   A escolha é da orientadora (item 1 da pauta). Documente que, com `"boneco"`, **nenhuma** contagem de comentários pode ser feature.

2. Definir em `modeling.py`:
   ```python
   FORBIDDEN_FEATURES = frozenset({
       "qtd_comentarios_coletados", "n_comments_collected",
       "total_comments_count", "comment_count_video",
       "likes", "like_count_video", "views", "view_count",
       "avg_comment_like_count", "avg_reply_count",
   })
   ```
   As duas últimas não estão no alvo, mas são medidas de engajamento usadas para prever engajamento — circulares e não informativas sobre a hipótese.

3. Escrever `assert_no_leakage(feature_columns)` que levanta `ValueError` se a interseção com `FORBIDDEN_FEATURES` for não-vazia. Chamar no início de `train_and_evaluate` e `compare_models`.

4. **Mediana só no treino.** Substituir o cálculo global por:
   ```python
   def fit_threshold(y_train) -> float: ...
   def apply_threshold(y, threshold) -> np.ndarray: ...
   ```
   Dentro de validação cruzada, o threshold é recalculado **em cada fold**, a partir do treino daquele fold. Nunca do conjunto completo.

**Teste** (`tests/test_no_leakage.py`): `assert_no_leakage` levanta em cada feature proibida; o threshold calculado num fold difere do global em dados sintéticos.

**Commit:** `fix: remove vazamento de alvo em features e no cálculo do threshold`

---

## Tarefa 3 — Restaurar as features e a higiene do sklearn

**Contexto:** o notebook base tinha 10 features e um `Pipeline` com imputação e escalonamento. A versão modular ficou com 4 features e sem pré-processamento. Regressão a desfazer.

**Fazer:**

1. Restaurar as features de sentimento removidas, calculadas em `feature_engineering.py`:
   - `sentiment_mean` — média de `sentiment_numeric` (score assinado: `+score` se positivo, `-score` se negativo, `0.0` se neutro)
   - `sentiment_std` — desvio-padrão de `sentiment_numeric`. **Mede polarização/controvérsia.** `fillna(0)` quando o vídeo tem 1 comentário.
   - `avg_text_length`, `avg_word_count`
   - `positive_ratio`, `negative_ratio`, `neutral_ratio`

2. Adicionar as variáveis de controle exigidas pelo notebook base (§14, item 3) e pelo protocolo:
   - `log_inscritos`, `idade_dias`, `duracao_seg`

3. Definir dois conjuntos explícitos:
   ```python
   METADATA_FEATURES = ["log_inscritos", "idade_dias", "duracao_seg", "avg_text_length", "avg_word_count"]
   SENTIMENT_FEATURES = ["sentiment_mean", "sentiment_std", "positive_ratio", "negative_ratio", "neutral_ratio"]
   FEATURE_COLUMNS = METADATA_FEATURES + SENTIMENT_FEATURES
   ```
   Essa separação é o que torna a H2 testável na Tarefa 5. Não a colapse.

4. Restaurar o pré-processamento do notebook base:
   ```python
   Pipeline([("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler())])
   ```
   dentro de um `ColumnTransformer`, dentro do `Pipeline` do modelo. O scaler **fica dentro** do pipeline para não vazar estatísticas do teste.

5. `random_state=42` em todo `train_test_split`, `RandomForest*`, `XGB*`, `KFold`, `StratifiedKFold`.

**Commit:** `refactor: restaura features de sentimento, controles e pipeline sklearn`

---

## Tarefa 4 — Baselines obrigatórios

**Contexto:** sem piso de comparação, nenhuma métrica é interpretável.

**Fazer:** em `compare_models`, as **três primeiras linhas** da tabela são:

1. `DummyRegressor(strategy="mean")`
2. `DummyClassifier(strategy="most_frequent")`
3. **Modelo só-metadados** — mesmo estimador, treinado apenas com `METADATA_FEATURES`

Depois vêm Linear/Logistic Regression, Random Forest e XGBoost, cada um em duas variantes: `metadata` e `metadata+sentiment`.

Adicionar `ROC-AUC` às métricas de classificação (Boneco §6.7 pede; hoje só há acurácia e F1).

**Guarda de N.** Restaurar e endurecer o aviso da orientadora:
- `n < 8` → `RuntimeError`. Não treine.
- `8 <= n < 50` → aviso gritante no log e coluna `confiavel=False` em toda linha da tabela de métricas.
- `n >= 50` → normal.

**Commit:** `feat: baselines dummy e só-metadados, ROC-AUC, guarda de tamanho amostral`

---

## Tarefa 5 — Testes das hipóteses H1, H2 e H3

**Contexto:** este é o núcleo científico do TCC, e hoje não existe. As três hipóteses do Boneco são inverificáveis com o código atual.

Criar `src/hypotheses.py`.

### H1 — proporção de positivos → maior engajamento

```python
def test_h1(df_final) -> dict:
    """Correlação de Spearman entre positive_ratio e engagement_rate.
    Retorna rho, p-valor, n, e IC bootstrap de 95%.
    Usar Spearman (não Pearson): engagement_rate é fortemente assimétrico.
    Reportar também a correlação parcial controlando log_inscritos.
    """
```
A correlação parcial não é opcional. Sem controlar o tamanho do canal, a correlação bruta mede popularidade do canal.

### H2 — sentimento melhora a predição

```python
def test_h2(df_final, n_splits=10) -> dict:
    """Compara metadata-only vs metadata+sentiment.
    Mesmos folds para os dois (StratifiedKFold com random_state fixo).
    Retorna: scores por fold de cada modelo, diferença média,
    e o resultado do Wilcoxon pareado (ver test_h3).
    """
```
Os folds devem ser **os mesmos** para os dois modelos. Emparelhamento é o que dá poder ao teste.

### H3 — impacto estatisticamente significativo

```python
def test_h3(h2_result) -> dict:
    """Wilcoxon signed-rank pareado sobre os scores por fold.
    Reportar estatística, p-valor e tamanho de efeito (r = Z / sqrt(N)).
    Complementar com permutation importance das SENTIMENT_FEATURES
    no modelo completo (n_repeats=30, random_state=42).
    """
```

**Regras de reporte, obrigatórias:**
- Reportar sempre o **tamanho do efeito**, não só o p-valor.
- Se `n < 50`, marcar todo resultado como exploratório.
- **Não** rodar os testes múltiplas vezes trocando parâmetros até dar significativo. Se for testar variantes, aplicar correção de Bonferroni e declarar quantos testes foram feitos.
- Um resultado nulo é um resultado. Se o sentimento não ajudar, isso se reporta com a mesma clareza.

**Teste:** com a fixture `effect_size=0.0`, `test_h3` deve devolver p > 0.05. Se devolver significativo, há bug — provavelmente vazamento.

**Commit:** `feat: testes estatísticos das hipóteses H1, H2 e H3`

---

## Tarefa 6 — Coleta escalável (sem coletar)

**Contexto:** a moldura amostral ainda não existe. Prepare a infraestrutura para quando existir.

**Fazer:**

1. `VIDEO_IDS` sai de `config.py`. Passa a ser lido de `data/videos_selecionados.csv` com colunas:
   `video_id, canal_id, canal_nome, inscritos, categoria, data_publicacao, estrato`
   Se o arquivo não existir ou estiver vazio: erro claro apontando para `PROTOCOLO_AMOSTRAGEM.md`. **Não** invente IDs de exemplo.

2. Cache em disco. `.cache/{video_id}.json` guarda a resposta crua da API. Se existe, não chame a API. Adicionar `.cache/` ao `.gitignore`.

3. Em `commentThreads.list`, trocar `order="relevance"` por `order="time"`. Comentário no código explicando: a ordenação por relevância retorna os mais curtidos, sistematicamente mais positivos, e essa positividade correlaciona com o alvo.

4. `MAX_COMMENTS_PER_VIDEO = 200` (valor do notebook base; hoje está 100).

5. Logar contadores de exclusão por critério (comentários desabilitados, < 30 comentários, views < 1000, etc.) num CSV `outputs/fluxo_amostral.csv`. Isso vira o diagrama de fluxo amostral do TCC.

**Testes:** mockar o cliente da API. Verificar que o cache evita a segunda chamada; que `order="time"` é passado; que arquivo ausente levanta erro instrutivo.

**Commit:** `feat: moldura amostral em CSV, cache de API, order=time`

---

## Tarefa 7 — RAG: congelado

**Não mexer.** A camada RAG (`src/rag/`) fica exatamente como está até a orientadora decidir (item 8 da pauta): cortar do escopo, ou transformar em contribuição real injetando valores SHAP no prompt.

Duas coisas apenas, ambas baratas:

1. Em `pipeline.py`, tornar a etapa 8 opcional via `config.RUN_RAG = False` (padrão). Hoje o índice FAISS é construído e descartado a cada execução, porque `main.py` não captura o retorno de `run_pipeline()`.
2. Adicionar ao topo de `rag/explainer.py`:
   ```python
   # AVISO METODOLÓGICO: este módulo gera explicações em linguagem natural
   # SEM acesso ao modelo de ML. São narrativas plausíveis, não explicações
   # fiéis à decisão do Random Forest. Não usar como evidência no TCC.
   # Ver Parcalabescu & Frank (ACL 2024) sobre fidelidade de explicações de LLM.
   ```

**Commit:** `chore: torna etapa RAG opcional e documenta limitação metodológica`

---

## Ordem de execução e critério de pronto

| # | Tarefa | Depende de | Pronto quando |
|---|---|---|---|
| 0 | Fixture sintética | — | 4 testes passam |
| 1 | Sentimento | 0 | Modelo real, emojis preservados, rótulos mapeados |
| 2 | Vazamento | 0 | `assert_no_leakage` ativo, threshold no fold |
| 3 | Features + sklearn | 1, 2 | 10 features, `METADATA`/`SENTIMENT` separados |
| 4 | Baselines | 3 | 3 baselines na tabela, guarda de N |
| 5 | Hipóteses | 4 | H1/H2/H3 rodam; `effect_size=0` não dá significativo |
| 6 | Coleta | — | Cache funciona, `order="time"`, CSV obrigatório |
| 7 | RAG | — | Etapa opcional, aviso no docstring |

**Só depois das tarefas 0–6 concluídas, com a moldura amostral aprovada pela orientadora, execute a coleta real.**

---

## O que NÃO fazer

- Não chamar a YouTube API. Não gerar `VIDEO_IDS` de exemplo.
- Não reintroduzir contagens de comentários, likes ou views como features.
- Não calcular a mediana sobre o dataset completo.
- Não inventar nomes de modelos do Hugging Face.
- Não implementar SHAP nem tocar no RAG antes da decisão da orientadora.
- Não "consertar" um teste que falha ajustando o limiar até passar. Se `test_no_false_positive` falhar, há vazamento — encontre-o.
- Não silenciar warnings de amostra pequena.
