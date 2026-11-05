# CLAUDE.md — Regras do projeto TCC_Engajamento

## Contexto do projeto

Este projeto implementa um pipeline de análise de sentimento e predição de engajamento em vídeos do YouTube.

A arquitetura principal está documentada em `ARCHITECTURE.md`.

O pipeline utiliza:

- Python
- pandas
- scikit-learn
- XGBoost
- Hugging Face Transformers
- análise de sentimento em comentários
- modelos de predição de engajamento
- camada RAG opcional

## Regras obrigatórias

1. Leia sempre:
   - `ARCHITECTURE.md`
   - `TAREFAS.md`
   - `PROTOCOLO_AMOSTRAGEM.md`

   antes de executar qualquer alteração.

2. Execute somente uma tarefa por sessão.

3. Antes de qualquer tarefa que altere múltiplos arquivos:
   - entre em plan mode;
   - apresente o plano;
   - aguarde aprovação.

4. Não execute tarefas fora da ordem definida em `TAREFAS.md`.

5. Nunca use dados reais de coleta antes da aprovação da moldura amostral.

6. Nunca chame a YouTube Data API durante as tarefas 0 a 6.

7. Use dados sintéticos para validar o pipeline quando indicado.

8. Nunca invente:
   - IDs de vídeos;
   - nomes de modelos Hugging Face;
   - resultados experimentais;
   - métricas.

9. Se uma instrução estiver ambígua ou conflitante, pare e pergunte.

## Testes

Após alterações:

```bash
python -m pytest -v
