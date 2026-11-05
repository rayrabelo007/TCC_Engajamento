# PROTOCOLO_AMOSTRAGEM.md — Protocolo de seleção amostral

## Objetivo

Definir previamente os critérios de seleção dos vídeos utilizados no estudo de predição de engajamento a partir de análise de sentimento de comentários do YouTube.

Nenhuma coleta real deve ser realizada antes da aprovação destes critérios.

---

## Estado atual

A moldura amostral ainda está em definição.

Durante as tarefas de desenvolvimento do pipeline:

- não utilizar dados reais;
- não chamar a YouTube Data API;
- utilizar apenas dados sintéticos para validação do código.

---

## Unidade de análise

A unidade principal de análise é o vídeo.

Cada vídeo terá:

- características do canal;
- características do conteúdo;
- comentários coletados;
- agregações de sentimento;
- métricas de engajamento.

---

## Critérios de inclusão

Devem ser definidos antes da coleta:

- período de publicação dos vídeos;
- categoria temática;
- quantidade mínima de visualizações;
- quantidade mínima de comentários;
- disponibilidade de comentários públicos;
- idioma predominante.

---

## Critérios de exclusão

Registrar previamente:

- vídeos sem comentários disponíveis;
- vídeos com comentários desativados;
- vídeos abaixo do limite mínimo de dados;
- vídeos fora do escopo temático.

---

## Conteúdo publicitário

A definição operacional de conteúdo publicitário deve ser decidida antes da coleta.

Possíveis critérios:

1. Identificação explícita:
   - hashtag #publi;
   - #publicidade;
   - indicação de parceria paga.

2. Declaração no texto da descrição do vídeo.

3. Curadoria manual baseada no conteúdo.

Cada alternativa possui potenciais vieses e deve ser justificada metodologicamente.

---

## Amostragem

A estratégia de amostragem deve definir:

- população-alvo;
- tamanho esperado da amostra;
- critérios de estratificação;
- distribuição por categoria/canal.

Essas decisões devem ser registradas antes da coleta.

---

## Reprodutibilidade

A seleção final deve gerar um arquivo:

data/videos_selecionados.csv

com as colunas:

- video_id
- canal_id
- canal_nome
- inscritos
- categoria
- data_publicacao
- estrato

Esse arquivo será a fonte única dos vídeos coletados.

---

## Regra metodológica

Nenhum resultado numérico do TCC deve ser produzido com dados sintéticos.

Dados sintéticos existem exclusivamente para:

- validar implementação;
- testar ausência de vazamento;
- verificar recuperação de efeitos conhecidos.

