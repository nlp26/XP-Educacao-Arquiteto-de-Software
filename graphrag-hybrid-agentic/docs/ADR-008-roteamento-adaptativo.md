# ADR-008 — Roteamento adaptativo em vez de uma estrategia unica

**Status**: aceito

## Contexto

E' tentador escolher "a melhor" estrategia e usa-la sempre. A avaliacao deste projeto
mostra por que isso e' errado: na tabela por tipo de pergunta
(`evidence/run_summary.md`), o BM25 puro tem o **melhor** nDCG@3 nas perguntas factuais
(1,000) e um dos piores nas tematicas (0,628); o Local Search lidera no agregado; o DRIFT
Search custa ~10x mais latencia que a busca hibrida e nao lidera em nenhuma coluna neste
corpus.

Nao existe estrategia dominante. Existe estrategia adequada ao tipo de pergunta — e o
tipo e' barato de estimar.

## Decisao

Classificar cada consulta em quatro tipos e rotear:

| Tipo | Estrategia | Justificativa |
|---|---|---|
| factual | `hybrid` | uma evidencia basta; grafo so adiciona custo |
| relacional | `local_search` | multi-hop entre entidades nomeadas |
| tematica | `global_search` | nenhum chunk isolado responde; os Community Reports respondem |
| comparativa | `drift_search` | precisa localizar duas regioes do grafo antes de comparar |

O classificador e' heuristico por padrao (marcadores casados por palavra inteira +
contagem de entidades ancoradas no grafo) e usa LLM quando ha backend. Quando o entity
linking nao ancora nada, `local_search` degrada explicitamente para `hybrid`.

O ciclo fecha com um critico que mede **fundamentacao** e **cobertura** separadamente —
sao modos de falha distintos: fundamentacao baixa indica extrapolacao do gerador,
cobertura baixa indica falha de recuperacao. Reprovando, o planejador **escala** (troca
de estrategia, amplia o top-k, decompoe a pergunta) em vez de repetir a mesma busca.

## Consequencias

**Positivas**
- A estrategia cara so e' paga quando o tipo de pergunta a justifica.
- Acuracia do roteador heuristico no conjunto de avaliacao: **94,4%** (17/18).
- O unico erro (`q08`) e' informativo: uma pergunta relacional sem marcador lexical e com
  uma unica entidade ancorada e' indistinguivel de uma factual sem ler o corpus.

**Negativas**
- Marcadores lexicais nao generalizam para outro dominio ou idioma sem revisao.
- O roteador e' um ponto unico de erro: rota errada custa uma iteracao inteira.
- No agregado, o modo roteado nao supera o `local_search` puro **neste corpus** — com 12
  documentos, as comunidades sao grandes demais para o Global Search ser preciso. O ganho
  do roteamento cresce com o tamanho do corpus; a honestidade exige registrar isso.

## Alternativas consideradas

| Alternativa | Por que nao |
|---|---|
| Sempre DRIFT | ~10x a latencia sem liderar nenhuma coluna neste corpus |
| Sempre hibrida | perde as perguntas tematicas e relacionais de dois saltos |
| Classificador treinado | exige dados rotulados; o heuristico ja acerta 94% aqui |
