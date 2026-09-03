# GraphRAG + Busca Hibrida Agentica

Implementacao de referencia, executavel e sem dependencias, das tecnicas de recuperacao
que definem o estado da pratica em 2026: **GraphRAG** (Local Search, Global Search,
DRIFT Search, LazyGraphRAG), **busca hibrida** (BM25 + denso, fusao RRF, reranking, MMR)
e a camada **agentica** que decide qual delas usar por consulta.

Sete estrategias compartilham a mesma assinatura e sao medidas na mesma bancada — com
qualidade **e custo** lado a lado, que e' onde a maioria das comparacoes de RAG falha.

```bash
python main.py "Como o padrao Outbox garante que o evento chegue ao Kafka?"
```

Sem `pip install`. Sem chave de API. Sem GPU. Python 3.11+ e a stdlib.

---

## Por que este exemplo existe

A pergunta que importa em 2026 nao e' mais "GraphRAG ou busca vetorial?" — e' **quando
cada uma vale o custo**. Este projeto responde isso com medicao, nao com opiniao:

| Estrategia | Recall@3 | nDCG@3 | MRR | Latencia (ms) | Chamadas LLM/consulta |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.796 | 0.836 | 0.935 | 0.0 | 0.0 |
| vector | 0.801 | 0.753 | 0.787 | 0.5 | 0.0 |
| hybrid | 0.769 | 0.802 | 0.917 | 1.8 | 0.0 |
| **local_search** | **0.824** | **0.841** | 0.917 | 3.3 | 0.0 |
| global_search | 0.755 | 0.765 | 0.917 | 2.5 | 0.0 |
| drift_search | 0.750 | 0.765 | 0.880 | 30.9 | 0.0 |
| lazy_graphrag | 0.782 | 0.825 | **0.944** | 3.2 | 0.0 |
| agentic (roteado) | 0.782 | 0.798 | 0.917 | 30.1 | 1.0 |

E, mais util que o agregado, a mesma medida **por tipo de pergunta**:

| Estrategia | factual | relacional | tematica | comparativa |
|---|---:|---:|---:|---:|
| bm25 | **1.000** | 0.769 | 0.628 | 0.947 |
| vector | 0.753 | **0.794** | 0.519 | 0.857 |
| hybrid | 0.848 | 0.763 | 0.638 | 0.947 |
| local_search | 0.988 | 0.773 | 0.638 | **0.967** |
| global_search | 0.756 | 0.763 | 0.539 | 0.947 |
| drift_search | 0.857 | 0.732 | 0.588 | 0.863 |
| lazy_graphrag | 0.940 | 0.746 | **0.695** | 0.947 |

> Numeros reais de `python run_evidence.py` (18 perguntas anotadas, corpus de 12
> documentos, backend `offline` deterministico). Reproduza e confira.

**O que isso mostra**, sem enfeite:

1. **Nao existe estrategia dominante.** BM25 puro e' imbativel em pergunta factual
   (nDCG 1.000) e um dos piores em pergunta tematica. Escolher uma unica estrategia para
   todo tipo de consulta e' perder de proposito — dai o roteador (ADR-008).
2. **DRIFT custa ~10x a latencia da busca hibrida** e nao lidera nenhuma coluna neste
   corpus. Qualidade sem custo ao lado e' meia medida.
3. **LazyGraphRAG entrega o melhor MRR** sem gerar nenhum Community Report — o trade-off
   "indexacao barata / consulta cara" aparece como numero, nao como afirmacao.
4. **O corpus e' pequeno de proposito** (19 text units): com `k=3`, cada consulta ja ve
   16% do corpus, entao os valores absolutos sao inflados. O que vale ler e' a **ordenacao
   relativa** e o custo. O ganho do Global Search cresce com o tamanho do corpus, porque
   comunidades so ficam informativas quando sao subconjuntos de verdade.

---

## Arquitetura

```
                              ┌──────────────────────────────────┐
   corpus/*.md ──────────────►│  INGESTAO  (grag/pipeline.py)    │
                              ├──────────────────────────────────┤
                              │ chunking por frase (text.py)     │
                              │ BM25          (retrievers.py)    │
                              │ vetores       (embeddings.py)    │
                              │ entidades+relacoes (extraction)  │
                              │ resolucao de entidades (graph)   │
                              │ comunidades → reports            │
                              └──────────────┬───────────────────┘
                                             │  Index (compartilhado)
                                             ▼
   pergunta ──►┌──────────────────────────────────────────────────────────┐
               │  AGENTE  (grag/agent.py)                                 │
               │                                                          │
               │   plan ──► retrieve ──► synthesize ──► critique          │
               │     ▲                                       │            │
               │     └───────── replan (escala) ◄────────────┘            │
               └───────────────────────┬──────────────────────────────────┘
                                       │ roteamento adaptativo
        ┌──────────────┬───────────────┼───────────────┬──────────────────┐
        ▼              ▼               ▼               ▼                  ▼
     factual       relacional       tematica       comparativa      (baselines)
     hybrid       local_search    global_search    drift_search    bm25 / vector
        │              │               │               │                  │
        └──────────────┴───────────────┴───────────────┴──────────────────┘
                                       │
                         ┌─────────────▼──────────────┐
                         │  BUSCA HIBRIDA (nucleo)    │
                         │  BM25 ─┐                   │
                         │  denso ─┼─► RRF ponderada  │
                         │  grafo ─┘   ↓              │
                         │        reranker → MMR      │
                         └────────────────────────────┘
```

Todo caminho passa pelo mesmo nucleo hibrido. O grafo nunca **substitui** a recuperacao
lexical/densa — ele entra como uma perna adicional na fusao e como bonus no reranker.
Consequencia pratica: quando o entity linking falha, a estrategia de grafo degrada
exatamente para a busca hibrida, nunca para algo pior.

### As sete estrategias

| Estrategia | O que faz | Quando usar |
|---|---|---|
| `bm25` | Okapi BM25 puro | baseline lexical; termos raros e exatos |
| `vector` | busca densa pura | baseline semantico; parafrase |
| `hybrid` | BM25 + denso → RRF → reranker → MMR | padrao para pergunta factual |
| `local_search` | ancora em entidades → Personalized PageRank → funde com hibrida | relacoes entre entidades nomeadas |
| `global_search` | map-reduce sobre Community Reports | perguntas tematicas do corpus inteiro |
| `drift_search` | primer global → perguntas de acompanhamento → locais iterativas | comparativa / multi-hop dificil |
| `lazy_graphrag` | BFS no grafo em tempo de consulta, sem Community Reports | custo de indexacao proximo de zero |

### O ciclo agentico

1. **Plan** — classifica a pergunta (factual / relacional / tematica / comparativa) e
   escolhe a estrategia. Heuristico por padrao (**94,4% de acuracia** no conjunto de
   avaliacao), por LLM quando ha backend.
2. **Retrieve** — executa a estrategia; com decomposicao, funde as rodadas por RRF
   **ponderada** (a pergunta original pesa o dobro de cada sub-pergunta).
3. **Synthesize** — gera a resposta e ancora cada frase ao trecho que a sustenta, mesmo
   quando o gerador nao cita.
4. **Critique** — mede **fundamentacao** (a resposta esta nos trechos citados?) e
   **cobertura** (a pergunta foi respondida por inteiro?) separadamente. Sao modos de
   falha distintos e pedem correcoes distintas.
5. **Replan** — reprovando, **escala**: troca de estrategia, amplia o top-k, gera
   sub-perguntas para as lacunas. Nunca repete a mesma busca.

---

## Como rodar

```bash
git clone https://github.com/nlp2026/xp-educacao-arquiteto-de-software.git
cd xp-educacao-arquiteto-de-software/graphrag-hybrid-agentic
```

```bash
# pergunta unica (o agente escolhe a estrategia)
python main.py "Qual a relacao entre CQRS e Event Sourcing?"

# forcar uma estrategia
python main.py --strategy drift_search "Compare DRIFT Search e LazyGraphRAG"

# ver as sete estrategias lado a lado na mesma pergunta
python main.py --compare "Como o Outbox garante a entrega no Kafka?"

# inspecionar o grafo, as entidades por grau e as comunidades
python main.py --graph

# arvore de spans da execucao
python main.py --trace "Por que o MMR importa mais em GraphRAG?"

# modo interativo
python main.py
```

```bash
# avaliacao comparativa
python -m evaluation.evaluate --k 3

# suite completa de evidencias (gera evidence/)
python run_evidence.py

# testes (stdlib, sem pytest)
python -m unittest discover -s tests -v
```

### Ligando um LLM de verdade

O nucleo nao muda — sao as mesmas estrategias com melhor extracao, melhores relatorios de
comunidade e sintese fluente:

```bash
# local, via Ollama
ollama pull qwen2.5:7b && ollama pull nomic-embed-text
GRAG_LLM=ollama GRAG_EMBEDDINGS=ollama python main.py "sua pergunta"

# via API Claude
export ANTHROPIC_API_KEY=...
GRAG_LLM=anthropic GRAG_MODEL=claude-sonnet-5 python main.py "sua pergunta"
```

Se o backend escolhido nao responder, a fabrica degrada para `offline` em vez de falhar.

---

## Estrutura

```
graphrag-hybrid-agentic/
├── main.py                   # CLI: consulta, comparacao, inspecao do grafo
├── run_evidence.py           # suite de evidencias → evidence/
├── requirements.txt          # tudo opcional; o nucleo e' stdlib
│
├── grag/
│   ├── config.py             # Settings: um experimento = um objeto
│   ├── text.py               # tokenizacao pt/en, stemmer sufixal, chunking por frase
│   ├── embeddings.py         # HashingEmbedder (offline) | OllamaEmbedder
│   ├── retrievers.py         # BM25, indice denso, RRF ponderada, convexa, reranker, MMR
│   ├── extraction.py         # entidades e relacoes (LLM ou gazetteer de duas passadas)
│   ├── graph.py              # grafo, resolucao de entidades, Label Propagation, PPR, BFS
│   ├── communities.py        # Community Reports (map do Global Search)
│   ├── strategies.py         # as sete estrategias
│   ├── agent.py              # roteador adaptativo, decomposicao, sintese, critico
│   ├── pipeline.py           # ingestao e o objeto Index compartilhado
│   ├── llm.py                # offline (deterministico) | ollama | anthropic
│   └── observability.py      # spans aninhados + contadores de custo (ponte para OTel)
│
├── corpus/                   # 12 documentos sobre arquitetura de software e RAG
├── evaluation/
│   ├── goldset.json          # 18 perguntas com relevancia graduada e tipo esperado
│   └── evaluate.py           # Recall@k, nDCG@k, MRR, latencia, custo, acuracia do roteador
├── tests/test_grag.py        # 24 testes (unittest)
├── docs/ADR-004..008         # decisoes de arquitetura
└── evidence/                 # artefatos gerados pela execucao
```

---

## Decisoes de arquitetura

| ADR | Decisao | Resumo |
|---|---|---|
| [004](docs/ADR-004-label-propagation.md) | Label Propagation no lugar do Leiden | deteccao deterministica sem dependencia C |
| [005](docs/ADR-005-llm-offline-deterministico.md) | LLM offline deterministico como padrao | avaliacao reproduzivel; mede recuperacao, nao fluencia |
| [006](docs/ADR-006-observabilidade-propria.md) | Observabilidade propria com ponte para OTel | spans e contadores de custo sem instalar coletor |
| [007](docs/ADR-007-fusao-rrf-ponderada.md) | RRF ponderada como fusao padrao | imune a escala; o peso resolve o modo de falha do DRIFT |
| [008](docs/ADR-008-roteamento-adaptativo.md) | Roteamento adaptativo | a estrategia cara so e' paga quando o tipo de pergunta justifica |

Continuam a numeracao dos ADRs #001–#003 do prototipo multiagente deste repositorio.

---

## Detalhes que costumam ser omitidos em exemplos de GraphRAG

- **Resolucao de entidades.** Sem ela o grafo fragmenta: "DRIFT" e "DRIFT Search" viram
  nos distintos e o grau de cada um cai pela metade. Tres regras conservadoras (flexao,
  prefixo contido, sigla) fundem 8 das 53 entidades brutas do corpus — inclusive
  `Command Query Responsibility Segregation` → `CQRS`.
- **Chunking em fronteira de frase.** Cortar em janela fixa de tokens quebra a relacao
  sujeito-predicado de que o extrator depende. Chunking ruim degrada o grafo antes de
  degradar o vetor.
- **MMR importa mais em GraphRAG.** A expansao pelo grafo traz varios trechos quase
  identicos sobre a mesma entidade, e o orcamento de contexto acaba antes de cobrir a
  pergunta. A varredura em `evidence/sweep.md` fixou `lambda = 0.85`.
- **Coverage nao pode punir o corpus.** Um termo da pergunta que nao existe em lugar
  nenhum do corpus nao pode ser "coberto"; conta-lo como lacuna dispara escalonamentos
  inuteis. O critico so cobra termos presentes no vocabulario indexado.
- **Custo de indexacao versus custo de consulta.** Sao contadores separados no tracer.
  E' a unica forma honesta de comparar GraphRAG classico com LazyGraphRAG.

## Limitacoes conhecidas

- **Corpus pequeno** (12 documentos, 19 text units): os valores absolutos sao inflados e
  o Global Search e' subestimado, porque as comunidades ficam grandes demais.
- **Busca densa exaustiva**: sem HNSW/IVF. A interface `DenseIndex` isola essa troca.
- **Marcadores de roteamento sao lexicais** e em pt/en: nao generalizam para outro
  dominio sem revisao.
- **`HashingEmbedder` nao e' um encoder treinado** — e' um stand-in deterministico com
  n-gramas de caractere. Para medir busca hibrida de verdade, ligue um encoder real.
- **O reranker e' baseado em features**, nao um cross-encoder. Em producao, use
  `bge-reranker-v2-m3` ou equivalente sobre os 20–200 primeiros candidatos.

## Caminho para producao

| Componente | Aqui | Em producao |
|---|---|---|
| Indice denso | busca exaustiva em memoria | Qdrant / pgvector / Elasticsearch (HNSW) |
| Embeddings | hashing determinista | encoder treinado (`nomic-embed-text`, Voyage, Cohere) |
| Reranker | scorer de features | cross-encoder (`bge-reranker-v2-m3`) |
| Comunidades | Label Propagation | Leiden (`leidenalg`), com resolucao ajustavel |
| Grafo | dicionarios em memoria | Neo4j / Memgraph / AGE |
| Extracao | gazetteer de duas passadas | LLM com saida estruturada + validacao de esquema |
| Observabilidade | spans proprios | OpenTelemetry + coletor (`bridge_to_otel()` ja existe) |

## Referencias

- [LazyGraphRAG: setting a new standard for quality and cost](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) — Microsoft Research
- [GraphSearch: An Agentic Deep Searching Workflow for Graph RAG](https://arxiv.org/pdf/2509.22009)
- [Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search](https://arxiv.org/pdf/2604.09666)
- [Hybrid Search for RAG: Combining BM25 and Dense Vector Search](https://denser.ai/blog/hybrid-search-for-rag/)
- [Hybrid Search: BM25, Vector & Reranking Reference 2026](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies](https://arxiv.org/html/2604.01733v1)
