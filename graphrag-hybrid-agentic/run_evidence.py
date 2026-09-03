#!/usr/bin/env python3
"""
Suite de evidencias: executa o sistema inteiro e materializa artefatos em `evidence/`.

Gera:
    execution_log.txt   log completo da execucao (o que foi perguntado e respondido)
    traces.json         arvore de spans + contadores de custo
    graph.json          grafo de conhecimento e comunidades detectadas
    evaluation.json     metricas por estrategia e por tipo de pergunta
    sweep.md            varredura de hiperparametros (lambda do MMR x fusao)
    run_summary.md      resumo legivel, pronto para anexar a um documento de arquitetura

    python run_evidence.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluate import (  # noqa: E402
    evaluate_agent,
    evaluate_strategy,
    load_goldset,
    render_by_kind,
    render_table,
)
from grag.agent import GraphRAGAgent  # noqa: E402
from grag.config import Settings  # noqa: E402
from grag.observability import get_tracer, reset_tracer  # noqa: E402
from grag.pipeline import build_index  # noqa: E402
from grag.strategies import STRATEGIES  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
EVIDENCE = os.path.join(ROOT, "evidence")

DEMO_QUERIES = [
    ("factual", "O que e' CQRS e o que ele separa?"),
    ("relacional", "Como o padrao Outbox garante que o evento chegue ao Kafka?"),
    ("tematica", "Quais sao os temas centrais do corpus?"),
    ("comparativa", "Compare DRIFT Search e LazyGraphRAG quanto ao custo de indexacao"),
    ("multi-hop", "Qual metrica o reranker melhora e por que ele nao conserta recuperacao ruim?"),
]

K = 3


class Log:
    """Escreve na tela e no arquivo ao mesmo tempo."""

    def __init__(self, path: str) -> None:
        self.handle = open(path, "w", encoding="utf-8")

    def __call__(self, line: str = "") -> None:
        print(line)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


def main() -> None:
    os.makedirs(EVIDENCE, exist_ok=True)
    log = Log(os.path.join(EVIDENCE, "execution_log.txt"))
    started = datetime.now(timezone.utc)

    settings = Settings()
    settings.retrieval.top_k = K

    log("=" * 78)
    log("SUITE DE EVIDENCIAS — GraphRAG + Busca Hibrida Agentica")
    log(f"inicio: {started.isoformat()}")
    log(f"config: llm={settings.llm_backend} embeddings={settings.embedding_backend} "
        f"fusao={settings.retrieval.fusion} mmr_lambda={settings.retrieval.mmr_lambda} top_k={K}")
    log("=" * 78)

    # ── 1. indexacao ─────────────────────────────────────────────────────
    reset_tracer()
    index_started = time.perf_counter()
    index = build_index(settings, corpus_dir=os.path.join(ROOT, "corpus"))
    index_ms = (time.perf_counter() - index_started) * 1000

    log("\n[1] INDEXACAO")
    log(f"    documentos ............ {len(index.documents)}")
    log(f"    text units ............ {len(index.chunks)}")
    log(f"    entidades ............. {len(index.graph.entities)}")
    log(f"    relacoes .............. {len(index.graph.relations)}")
    log(f"    comunidades ........... {index.graph.stats()['communities']}")
    log(f"    custo de indexacao .... {index.index_cost}")
    log(f"    tempo ................. {index_ms:.0f} ms")

    with open(os.path.join(EVIDENCE, "graph.json"), "w", encoding="utf-8") as handle:
        json.dump(index.graph.to_dict(), handle, ensure_ascii=False, indent=2)

    log("\n    Comunidades de nivel 0 (top 5 por grau agregado):")
    for community in sorted(index.graph.communities.get(0, []), key=lambda c: -c.rank)[:5]:
        log(f"      [{community.community_id}] {community.title}")
        log(f"          membros: {', '.join(community.members[:8])}")

    # ── 2. consultas agenticas ───────────────────────────────────────────
    agent = GraphRAGAgent(index, settings)
    log("\n[2] CONSULTAS AGENTICAS (plan -> retrieve -> synthesize -> critique)")
    answers = []
    for kind, query in DEMO_QUERIES:
        result = agent.answer(query)
        answers.append(result.to_dict())
        log("\n" + "-" * 78)
        log(f"    [{kind}] {query}")
        log(f"    plano ......... tipo={result.plan.kind} estrategia={result.plan.strategy} ({result.plan.reason})")
        for subquery in result.plan.subqueries:
            log(f"                    └─ {subquery}")
        for step in result.retrieval.steps:
            log(f"    passo ......... {step}")
        log(f"    resposta ...... {result.answer[:400]}")
        log(f"    fontes ........ {', '.join(c['chunk_id'] for c in result.citations)}")
        log(
            f"    critico ....... {'APROVADO' if result.critique.passed else 'REPROVADO'} "
            f"fundamentacao={result.critique.groundedness:.2f} "
            f"cobertura={result.critique.coverage:.2f} iteracoes={result.iterations}"
        )

    with open(os.path.join(EVIDENCE, "answers.json"), "w", encoding="utf-8") as handle:
        json.dump(answers, handle, ensure_ascii=False, indent=2)

    # Traces sao exportados aqui, antes da avaliacao: o que interessa e' a arvore
    # da indexacao e das consultas demonstrativas. A varredura seguinte gera
    # milhares de spans repetidos que so inflariam o artefato.
    tracer = get_tracer()
    spans_exportados = len(tracer.spans)
    contadores = tracer.snapshot_counters()
    tracer.export_json(os.path.join(EVIDENCE, "traces.json"))

    # ── 3. avaliacao comparativa ─────────────────────────────────────────
    log("\n[3] AVALIACAO COMPARATIVA")
    queries = load_goldset()
    rows = [evaluate_strategy(name, index, settings, queries, K) for name in STRATEGIES]
    rows.append(evaluate_agent(index, settings, queries, K))
    kinds = list(dict.fromkeys(q["kind"] for q in queries))

    table = render_table(rows, K)
    by_kind = render_by_kind(rows, K, kinds)
    log("\n" + table)
    log("\n" + by_kind)
    agent_row = rows[-1]
    log(
        f"\n    acuracia do roteador: {agent_row['routing_accuracy']:.1%} "
        f"| iteracoes medias: {agent_row['avg_iterations']:.2f}"
    )

    with open(os.path.join(EVIDENCE, "evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump({"k": K, "rows": rows}, handle, ensure_ascii=False, indent=2)

    # ── 4. varredura de hiperparametros ──────────────────────────────────
    log("\n[4] VARREDURA DE HIPERPARAMETROS (lambda do MMR x metodo de fusao)")
    sweep_lines = [
        "# Varredura de hiperparametros",
        "",
        f"nDCG@{K} medio no conjunto de avaliacao ({len(queries)} perguntas).",
        "",
        "| fusao | lambda MMR | hybrid | local_search | lazy_graphrag |",
        "|---|---:|---:|---:|---:|",
    ]
    for fusion in ("rrf", "convex"):
        for lambda_ in (0.5, 0.7, 0.85, 1.0):
            variant = Settings()
            variant.retrieval.top_k = K
            variant.retrieval.fusion = fusion
            variant.retrieval.mmr_lambda = lambda_
            cells = [
                f"{evaluate_strategy(name, index, variant, queries, K)['ndcg@k']:.3f}"
                for name in ("hybrid", "local_search", "lazy_graphrag")
            ]
            line = f"| {fusion} | {lambda_} | " + " | ".join(cells) + " |"
            sweep_lines.append(line)
            log("    " + line)

    with open(os.path.join(EVIDENCE, "sweep.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(sweep_lines) + "\n")

    # ── 5. resumo ────────────────────────────────────────────────────────
    log(f"\n[5] OBSERVABILIDADE — {spans_exportados} spans exportados para traces.json")
    log(f"    contadores no momento da exportacao: {contadores}")
    log(f"    spans totais desta execucao (com avaliacao e varredura): {len(get_tracer().spans)}")

    finished = datetime.now(timezone.utc)
    summary = f"""# Resumo da execucao

- **Data (UTC)**: {finished.isoformat(timespec='seconds')}
- **Duracao total**: {(finished - started).total_seconds():.1f}s
- **Backend de LLM**: `{index.llm.name}` | **embeddings**: `{settings.embedding_backend}`
- **Corpus**: {len(index.documents)} documentos, {len(index.chunks)} text units
- **Grafo**: {len(index.graph.entities)} entidades, {len(index.graph.relations)} relacoes, comunidades {index.graph.stats()['communities']}
- **Custo de indexacao**: {index.index_cost}

## Comparacao de estrategias (k={K})

{table}

## Por tipo de pergunta

{by_kind}

- Acuracia do roteador adaptativo: **{agent_row['routing_accuracy']:.1%}**
- Iteracoes medias do ciclo agentico: **{agent_row['avg_iterations']:.2f}**

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
"""
    with open(os.path.join(EVIDENCE, "run_summary.md"), "w", encoding="utf-8") as handle:
        handle.write(summary)

    log(f"\nArtefatos escritos em {EVIDENCE}/")
    log(f"fim: {finished.isoformat()}  (duracao {(finished - started).total_seconds():.1f}s)")
    log.close()


if __name__ == "__main__":
    main()
