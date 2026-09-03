# Resumo da execucao

- **Data (UTC)**: 2026-09-03T03:18:13+00:00
- **Duracao total**: 2.6s
- **Backend de LLM**: `offline` | **embeddings**: `hashing`
- **Corpus**: 12 documentos, 19 text units
- **Grafo**: 45 entidades, 54 relacoes, comunidades {0: 17, 1: 13}
- **Custo de indexacao**: {'llm_calls_indexing': 0.0, 'chunks': 19.0, 'community_reports': 30.0}

## Comparacao de estrategias (k=3)

| Estrategia | Recall@3 | nDCG@3 | MRR | Latencia (ms) | Chamadas LLM/consulta |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.796 | 0.836 | 0.935 | 0.0 | 0.0 |
| vector | 0.801 | 0.753 | 0.787 | 0.5 | 0.0 |
| hybrid | 0.769 | 0.802 | 0.917 | 1.8 | 0.0 |
| local_search | 0.824 | 0.841 | 0.917 | 3.3 | 0.0 |
| global_search | 0.755 | 0.765 | 0.917 | 2.4 | 0.0 |
| drift_search | 0.750 | 0.765 | 0.880 | 29.0 | 0.0 |
| lazy_graphrag | 0.782 | 0.825 | 0.944 | 3.2 | 0.0 |
| agentic (roteado) | 0.782 | 0.798 | 0.917 | 29.7 | 1.0 |

## Por tipo de pergunta

nDCG@3 por tipo de pergunta

| Estrategia | factual | relacional | tematica | comparativa |
|---|---:|---:|---:|---:|
| bm25 | 1.000 | 0.769 | 0.628 | 0.947 |
| vector | 0.753 | 0.794 | 0.519 | 0.857 |
| hybrid | 0.848 | 0.763 | 0.638 | 0.947 |
| local_search | 0.988 | 0.773 | 0.638 | 0.967 |
| global_search | 0.756 | 0.763 | 0.539 | 0.947 |
| drift_search | 0.857 | 0.732 | 0.588 | 0.863 |
| lazy_graphrag | 0.940 | 0.746 | 0.695 | 0.947 |
| agentic (roteado) | 0.848 | 0.793 | 0.539 | 0.949 |

- Acuracia do roteador adaptativo: **94.4%**
- Iteracoes medias do ciclo agentico: **1.00**

## Artefatos gerados

| Arquivo | Conteudo |
|---|---|
| `execution_log.txt` | log completo da execucao |
| `traces.json` | spans aninhados e contadores de custo |
| `graph.json` | entidades, relacoes e comunidades |
| `answers.json` | respostas agenticas com plano, citacoes e critica |
| `evaluation.json` | metricas brutas por estrategia e por consulta |
| `sweep.md` | varredura de lambda do MMR x metodo de fusao |

> Reprodutivel: o backend `offline` e' deterministico, entao esta execucao
> produz os mesmos numeros em qualquer maquina com Python 3.11+.
