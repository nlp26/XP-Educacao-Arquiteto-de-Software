#!/usr/bin/env python3
"""
CLI do exemplo GraphRAG + Busca Hibrida Agentica.

    python main.py "Como o Outbox garante a entrega no Kafka?"
    python main.py --strategy drift_search "Compare DRIFT e LazyGraphRAG"
    python main.py --compare "Qual a relacao entre CQRS e Event Sourcing?"
    python main.py --graph          # inspeciona o grafo e as comunidades
    python main.py                  # modo interativo
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grag.agent import GraphRAGAgent
from grag.config import Settings
from grag.observability import get_tracer, reset_tracer
from grag.pipeline import build_index
from grag.strategies import STRATEGIES, run_strategy

ROOT = os.path.dirname(os.path.abspath(__file__))


def print_answer(result, *, show_trace: bool = False) -> None:
    print("\n" + "=" * 78)
    print(f"PERGUNTA   {result.query}")
    print(f"PLANO      tipo={result.plan.kind}  estrategia={result.plan.strategy}  ({result.plan.reason})")
    if result.plan.subqueries:
        for subquery in result.plan.subqueries:
            print(f"           └─ sub-pergunta: {subquery}")
    print("-" * 78)
    print(result.answer)
    print("-" * 78)
    print("FONTES")
    for citation in result.citations:
        print(f"  [{citation['marker']}] {citation['chunk_id']}  ({citation['title']})  score={citation['score']}")
    verdict = result.critique
    status = "APROVADO" if verdict.passed else "REPROVADO"
    print(
        f"CRITICO    {status}  fundamentacao={verdict.groundedness:.2f} "
        f"cobertura={verdict.coverage:.2f}  iteracoes={result.iterations}"
    )
    if verdict.gaps:
        print(f"           lacunas: {', '.join(verdict.gaps)}")
    for step in result.retrieval.steps:
        print(f"           · {step}")
    if show_trace:
        print("-" * 78)
        print(get_tracer().render())
    print("=" * 78)


def compare_strategies(index, query: str, settings: Settings) -> None:
    print(f"\nComparando estrategias para: {query}\n")
    header = f"| {'estrategia':<15} | {'top chunks':<52} | ms |"
    print(header)
    print("|" + "-" * 17 + "|" + "-" * 54 + "|----|")
    import time

    for name in STRATEGIES:
        started = time.perf_counter()
        retrieval = run_strategy(name, index, query, settings)
        elapsed = (time.perf_counter() - started) * 1000
        top = ", ".join(retrieval.chunk_ids[:3])
        print(f"| {name:<15} | {top:<52.52} | {elapsed:>4.0f} |")


def inspect_graph(index) -> None:
    graph = index.graph
    print("\nGRAFO DE CONHECIMENTO")
    print(f"  {graph.stats()}")
    print("\nEntidades por grau:")
    for name in sorted(graph.entities, key=lambda n: (-graph.degree(n), n))[:12]:
        entity = graph.entities[name]
        print(f"  {graph.degree(name):>5.1f}  {name:<28} ({entity.type}, {entity.mentions} mencoes)")
    print("\nComunidades (nivel 0):")
    for community in sorted(graph.communities.get(0, []), key=lambda c: -c.rank)[:6]:
        print(f"  [{community.community_id}] {community.title}")
        print(f"      membros: {', '.join(community.members[:8])}")
        if community.summary:
            print(f"      resumo:  {community.summary[:150]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG + busca hibrida agentica")
    parser.add_argument("query", nargs="*", help="pergunta (vazio = modo interativo)")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), help="forcar uma estrategia")
    parser.add_argument("--compare", action="store_true", help="rodar todas as estrategias lado a lado")
    parser.add_argument("--graph", action="store_true", help="inspecionar grafo e comunidades")
    parser.add_argument("--trace", action="store_true", help="imprimir a arvore de spans")
    parser.add_argument("--corpus", default=os.path.join(ROOT, "corpus"))
    parser.add_argument("--llm", default=None, help="offline | ollama | anthropic")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    settings = Settings()
    if args.llm:
        settings.llm_backend = args.llm
    if args.top_k:
        settings.retrieval.top_k = args.top_k

    index = build_index(settings, corpus_dir=args.corpus)
    print(
        f"Indice pronto: {len(index.documents)} documentos, {len(index.chunks)} chunks, "
        f"{len(index.graph.entities)} entidades, {len(index.graph.relations)} relacoes "
        f"| LLM={index.llm.name}"
    )

    if args.graph:
        inspect_graph(index)
        if not args.query:
            return

    agent = GraphRAGAgent(index, settings)
    query = " ".join(args.query).strip()

    if query:
        if args.compare:
            compare_strategies(index, query, settings)
            return
        reset_tracer()
        print_answer(agent.answer(query, force_strategy=args.strategy), show_trace=args.trace)
        return

    print("\nModo interativo. Ctrl-C ou linha vazia para sair.\n")
    while True:
        try:
            line = input("pergunta> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        reset_tracer()
        print_answer(agent.answer(line, force_strategy=args.strategy), show_trace=args.trace)


if __name__ == "__main__":
    main()
