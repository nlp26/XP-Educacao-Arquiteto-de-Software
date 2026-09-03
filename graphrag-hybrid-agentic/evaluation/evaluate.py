"""
Bancada de avaliacao: compara estrategias na mesma consulta, com custo junto.

Metricas:
    Recall@k   a evidencia entrou na lista?      (cobertura)
    nDCG@k     entrou em posicao alta?           (ordenacao, relevancia graduada)
    MRR        onde caiu o primeiro acerto?      (primeira impressao)
    latencia   ms por consulta                   (custo de tempo)
    llm_calls  chamadas por consulta             (custo economico)

Ler qualidade sem custo e' o erro classico de avaliacao de RAG: DRIFT quase
sempre ganha em nDCG, e quase sempre perde quando o custo entra na conta.

Uso:
    python -m evaluation.evaluate                 # todas as estrategias
    python -m evaluation.evaluate --strategy hybrid --k 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grag.agent import GraphRAGAgent  # noqa: E402
from grag.config import Settings  # noqa: E402
from grag.observability import get_tracer  # noqa: E402
from grag.pipeline import build_index  # noqa: E402
from grag.strategies import STRATEGIES, run_strategy  # noqa: E402

GOLDSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldset.json")


def load_goldset(path: str = GOLDSET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["queries"]


def recall_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    gold = {cid for cid, grade in relevance.items() if grade > 0}
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        relevance.get(chunk_id, 0) / math.log2(position + 1)
        for position, chunk_id in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(position + 1) for position, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(retrieved: list[str], relevance: dict[str, int]) -> float:
    for position, chunk_id in enumerate(retrieved, start=1):
        if relevance.get(chunk_id, 0) > 0:
            return 1.0 / position
    return 0.0


def evaluate_strategy(name: str, index, settings: Settings, queries: list[dict], k: int) -> dict:
    tracer = get_tracer()
    recalls, ndcgs, rrs, latencies, calls = [], [], [], [], []
    per_query: list[dict] = []

    for item in queries:
        tracer.reset_counters(prefix="llm.")
        started = time.perf_counter()
        retrieval = run_strategy(name, index, item["query"], settings)
        elapsed = (time.perf_counter() - started) * 1000
        retrieved = retrieval.chunk_ids

        recall = recall_at_k(retrieved, item["relevance"], k)
        ndcg = ndcg_at_k(retrieved, item["relevance"], k)
        rr = reciprocal_rank(retrieved, item["relevance"])
        recalls.append(recall)
        ndcgs.append(ndcg)
        rrs.append(rr)
        latencies.append(elapsed)
        calls.append(tracer.snapshot_counters().get("llm.calls", 0.0))
        per_query.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "recall": round(recall, 3),
                "ndcg": round(ndcg, 3),
                "rr": round(rr, 3),
            }
        )

    n = len(queries) or 1
    return {
        "strategy": name,
        "recall@k": sum(recalls) / n,
        "ndcg@k": sum(ndcgs) / n,
        "mrr": sum(rrs) / n,
        "latency_ms": sum(latencies) / n,
        "llm_calls": sum(calls) / n,
        "per_query": per_query,
    }


def evaluate_agent(index, settings: Settings, queries: list[dict], k: int) -> dict:
    """Avalia o modo agentico e, de quebra, a acuracia do roteador."""
    agent = GraphRAGAgent(index, settings)
    tracer = get_tracer()
    recalls, ndcgs, rrs, latencies, calls, iterations = [], [], [], [], [], []
    routed_ok = 0
    routing: list[dict] = []
    per_query: list[dict] = []

    for item in queries:
        tracer.reset_counters(prefix="llm.")
        started = time.perf_counter()
        result = agent.answer(item["query"])
        elapsed = (time.perf_counter() - started) * 1000
        retrieved = result.retrieval.chunk_ids

        recall = recall_at_k(retrieved, item["relevance"], k)
        ndcg = ndcg_at_k(retrieved, item["relevance"], k)
        rr = reciprocal_rank(retrieved, item["relevance"])
        recalls.append(recall)
        ndcgs.append(ndcg)
        rrs.append(rr)
        per_query.append(
            {
                "id": item["id"],
                "kind": item["kind"],
                "recall": round(recall, 3),
                "ndcg": round(ndcg, 3),
                "rr": round(rr, 3),
            }
        )
        latencies.append(elapsed)
        calls.append(tracer.snapshot_counters().get("llm.calls", 0.0))
        iterations.append(result.iterations)
        correct = result.plan.kind == item["kind"]
        routed_ok += int(correct)
        routing.append(
            {
                "id": item["id"],
                "expected": item["kind"],
                "predicted": result.plan.kind,
                "strategy": result.plan.strategy,
                "correct": correct,
            }
        )

    n = len(queries) or 1
    return {
        "strategy": "agentic (roteado)",
        "recall@k": sum(recalls) / n,
        "ndcg@k": sum(ndcgs) / n,
        "mrr": sum(rrs) / n,
        "latency_ms": sum(latencies) / n,
        "llm_calls": sum(calls) / n,
        "routing_accuracy": routed_ok / n,
        "avg_iterations": sum(iterations) / n,
        "routing": routing,
        "per_query": per_query,
    }


def render_by_kind(rows: list[dict], k: int, kinds: list[str]) -> str:
    """
    Media por tipo de pergunta.

    E' aqui que a comparacao fica util: no agregado o BM25 e' dificil de bater
    num corpus pequeno, mas a vantagem do grafo aparece concentrada nos tipos
    relacional e tematico — que e' exatamente a razao de existir do roteador.
    """
    header = "| Estrategia | " + " | ".join(kinds) + " |\n|---|" + "---:|" * len(kinds)
    lines = [f"nDCG@{k} por tipo de pergunta\n", header]
    for row in rows:
        cells = []
        for kind in kinds:
            values = [q["ndcg"] for q in row.get("per_query", []) if q.get("kind") == kind]
            cells.append(f"{sum(values) / len(values):.3f}" if values else "—")
        lines.append(f"| {row['strategy']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_table(rows: list[dict], k: int) -> str:
    header = (
        f"| Estrategia | Recall@{k} | nDCG@{k} | MRR | Latencia (ms) | Chamadas LLM/consulta |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['recall@k']:.3f} | {row['ndcg@k']:.3f} | "
            f"{row['mrr']:.3f} | {row['latency_ms']:.1f} | {row['llm_calls']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliacao comparativa das estrategias de recuperacao")
    parser.add_argument("--k", type=int, default=6, help="profundidade das metricas (default: 6)")
    parser.add_argument("--strategy", help="avaliar apenas uma estrategia")
    parser.add_argument("--corpus", default=None, help="diretorio do corpus")
    parser.add_argument("--json", dest="json_out", default=None, help="salvar resultado bruto em JSON")
    args = parser.parse_args()

    settings = Settings()
    settings.retrieval.top_k = args.k
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_dir = args.corpus or os.path.join(root, settings.corpus_dir)

    index = build_index(settings, corpus_dir=corpus_dir)
    queries = load_goldset()

    names = [args.strategy] if args.strategy else list(STRATEGIES)
    rows = [evaluate_strategy(name, index, settings, queries, args.k) for name in names]
    agent_row = evaluate_agent(index, settings, queries, args.k) if not args.strategy else None
    if agent_row:
        rows.append(agent_row)

    print(f"\nCorpus: {index.stats()['documents']} documentos, {index.stats()['chunks']} chunks")
    print(f"Grafo:  {index.graph.stats()}")
    print(f"Custo de indexacao: {index.index_cost}\n")
    print(render_table(rows, args.k))
    kinds = list(dict.fromkeys(q["kind"] for q in queries))
    print()
    print(render_by_kind(rows, args.k, kinds))
    if agent_row:
        print(
            f"\nAcuracia do roteador: {agent_row['routing_accuracy']:.1%} "
            f"| iteracoes medias: {agent_row['avg_iterations']:.2f}"
        )
        errors = [r for r in agent_row["routing"] if not r["correct"]]
        if errors:
            print("Erros de roteamento:")
            for error in errors:
                print(f"  {error['id']}: esperado={error['expected']} previsto={error['predicted']}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump({"k": args.k, "rows": rows}, handle, ensure_ascii=False, indent=2)
        print(f"\nResultado bruto salvo em {args.json_out}")


if __name__ == "__main__":
    main()
