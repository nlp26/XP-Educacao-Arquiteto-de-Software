"""
Estrategias de recuperacao — o coracao do exemplo.

Sete estrategias com a mesma assinatura `(index, query, settings) -> Retrieval`,
o que permite compara-las na mesma bancada:

    bm25            baseline lexical
    vector          baseline denso
    hybrid          BM25 + denso -> RRF -> reranker -> MMR
    local_search    ancoragem em entidades + Personalized PageRank (GraphRAG)
    global_search   map-reduce sobre Community Reports (GraphRAG)
    drift_search    primer global -> perguntas de acompanhamento -> local (iterativo)
    lazy_graphrag   sem relatorios: BFS no grafo em tempo de consulta

O grafo entra sempre como *sinal adicional* na fusao, nunca substituindo a
recuperacao lexical/densa. Essa e' a licao pratica de 2026: GraphRAG melhora
perguntas relacionais e tematicas, e piora perguntas factuais simples quando
usado sozinho — por isso existe o roteador em `agent.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .graph import Community
from .observability import get_tracer
from .pipeline import Index
from .retrievers import FeatureReranker, Scored, fuse, mmr
from .text import tokenize

_reranker = FeatureReranker()


@dataclass
class Retrieval:
    """Saida uniforme das estrategias — o que o gerador e o critico consomem."""

    strategy: str
    context: list[Scored] = field(default_factory=list)
    seed_entities: list[str] = field(default_factory=list)
    expanded_entities: list[str] = field(default_factory=list)
    communities: list[Community] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.context]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "chunks": [
                {"chunk_id": c.chunk_id, "score": round(c.score, 4), "provenance": c.provenance}
                for c in self.context
            ],
            "seed_entities": self.seed_entities,
            "expanded_entities": self.expanded_entities,
            "communities": [c.community_id for c in self.communities],
            "subqueries": self.subqueries,
            "steps": self.steps,
        }


# ── baselines ─────────────────────────────────────────────────────────────

def bm25_search(index: Index, query: str, settings: Settings) -> Retrieval:
    with get_tracer().span("retrieval.bm25"):
        results = index.bm25.search(query, top_n=settings.retrieval.top_k)
    return Retrieval("bm25", context=results, steps=["busca lexical pura"])


def vector_search(index: Index, query: str, settings: Settings) -> Retrieval:
    with get_tracer().span("retrieval.vector"):
        results = index.dense.search(query, top_n=settings.retrieval.top_k)
    return Retrieval("vector", context=results, steps=["busca densa pura"])


# ── busca hibrida ─────────────────────────────────────────────────────────

def hybrid_search(
    index: Index,
    query: str,
    settings: Settings,
    *,
    graph_bonus: dict[str, float] | None = None,
    extra_legs: dict[str, list[Scored]] | None = None,
    leg_weights: dict[str, float] | None = None,
    query_entities: list[str] | None = None,
    top_k: int | None = None,
) -> Retrieval:
    """
    Pipeline hibrido completo. `graph_bonus` e `extra_legs` sao os ganchos por
    onde as estrategias de grafo injetam o sinal estrutural sem duplicar codigo.
    """
    cfg = settings.retrieval
    top_k = top_k or cfg.top_k
    tracer = get_tracer()
    with tracer.span("retrieval.hybrid", fusion=cfg.fusion, top_k=top_k) as span:
        legs: dict[str, list[Scored]] = {
            "bm25": index.bm25.search(query, top_n=cfg.candidates_per_leg),
            "dense": index.dense.search(query, top_n=cfg.candidates_per_leg),
        }
        if extra_legs:
            legs.update(extra_legs)
        fused = fuse(legs, method=cfg.fusion, k=cfg.rrf_k, alpha=cfg.convex_alpha, weights=leg_weights)
        reranked = _reranker.rerank(
            query,
            fused,
            index.chunks,
            top_n=cfg.rerank_top_n,
            graph_bonus=graph_bonus,
            query_entities=query_entities or [],
        )
        selected = mmr(reranked, index.dense, top_k=top_k, lambda_=cfg.mmr_lambda)
        span.set_attribute("candidates", len(fused))
        span.set_attribute("legs", ",".join(legs))
    steps = [f"pernas: {', '.join(legs)}", f"fusao: {cfg.fusion}", "reranker + MMR"]
    return Retrieval("hybrid", context=selected, steps=steps, seed_entities=query_entities or [])


# ── GraphRAG: Local Search ────────────────────────────────────────────────

def local_search(index: Index, query: str, settings: Settings, *, top_k: int | None = None) -> Retrieval:
    """
    Ancora a pergunta em entidades, expande com Personalized PageRank e usa a
    massa de PPR como bonus no reranqueamento.

    Diferenca em relacao a "recuperar os chunks dos vizinhos": aqui o grafo nao
    decide o resultado, ele *inclina* a fusao. Quando o entity linking falha, a
    estrategia degrada exatamente para a busca hibrida — nunca para pior.
    """
    cfg = settings.graph
    tracer = get_tracer()
    with tracer.span("retrieval.local_search") as span:
        seeds = index.graph.link_entities(query)
        ranks = index.graph.personalized_pagerank(
            seeds, alpha=cfg.ppr_alpha, iterations=cfg.ppr_iterations
        )
        expanded = [name for name, mass in list(ranks.items())[:12] if mass > 0]
        span.set_attribute("seeds", ",".join(seeds[:5]) or "none")
        span.set_attribute("expanded", len(expanded))

        graph_bonus: dict[str, float] = {}
        max_mass = max(ranks.values(), default=0.0) or 1.0
        for name in expanded:
            entity = index.graph.entities.get(name)
            if not entity:
                continue
            normalized = ranks[name] / max_mass
            for chunk_id in entity.chunks:
                graph_bonus[chunk_id] = max(graph_bonus.get(chunk_id, 0.0), normalized)

        graph_leg = [
            Scored(chunk_id, bonus, {"leg": "graph"})
            for chunk_id, bonus in sorted(graph_bonus.items(), key=lambda kv: -kv[1])
        ][: settings.retrieval.candidates_per_leg]

        retrieval = hybrid_search(
            index,
            query,
            settings,
            graph_bonus=graph_bonus,
            extra_legs={"graph": graph_leg} if graph_leg else None,
            query_entities=expanded,
            top_k=top_k,
        )

    retrieval.strategy = "local_search"
    retrieval.seed_entities = seeds
    retrieval.expanded_entities = expanded
    retrieval.steps = [
        f"entity linking: {', '.join(seeds) or 'nenhuma entidade casada'}",
        f"PPR expandiu para {len(expanded)} entidades",
        *retrieval.steps,
    ]
    return retrieval


# ── GraphRAG: Global Search ───────────────────────────────────────────────

def global_search(
    index: Index,
    query: str,
    settings: Settings,
    *,
    level: int = 0,
    top_k: int | None = None,
) -> Retrieval:
    """
    Map-reduce sobre Community Reports.

    MAP: seleciona os relatorios relevantes e extrai pontos-chave com score.
    REDUCE: reune os text units que sustentam os pontos-chave mais bem
    pontuados. Perguntas tematicas ("quais os temas do corpus") nao casam com
    nenhum chunk isolado, mas casam com o resumo do subgrafo.
    """
    top_k = top_k or settings.retrieval.top_k
    tracer = get_tracer()
    with tracer.span("retrieval.global_search", level=level) as span:
        if index.community_index is None:
            return hybrid_search(index, query, settings, top_k=top_k)

        candidates = index.community_index.search(query, top_n=settings.graph.max_community_reports * 3)
        communities: list[Community] = []
        for candidate in candidates:
            community = index.community_by_id.get(candidate.chunk_id)
            if community is not None and community.level == level:
                communities.append(community)
            if len(communities) >= settings.graph.max_community_reports:
                break
        if not communities:  # nivel vazio: cai para o nivel 0
            communities = [
                index.community_by_id[c.chunk_id]
                for c in candidates[: settings.graph.max_community_reports]
                if c.chunk_id in index.community_by_id
            ]

        # MAP: pontuar cada relatorio pela cobertura dos termos da consulta
        query_terms = set(tokenize(query))
        key_points: list[tuple[float, Community]] = []
        for community in communities:
            report_terms = set(tokenize(community.report_text))
            coverage = len(query_terms & report_terms) / (len(query_terms) or 1)
            score = coverage + 0.15 * len(community.members) / 10
            key_points.append((score, community))
        key_points.sort(key=lambda item: -item[0])

        # REDUCE: text units de suporte, fundidos com a perna hibrida
        support: dict[str, float] = {}
        for score, community in key_points:
            for chunk_id in community.chunk_ids:
                support[chunk_id] = max(support.get(chunk_id, 0.0), score)
        support_leg = [
            Scored(chunk_id, score, {"leg": "community"})
            for chunk_id, score in sorted(support.items(), key=lambda kv: -kv[1])
        ][: settings.retrieval.candidates_per_leg]

        retrieval = hybrid_search(
            index,
            query,
            settings,
            graph_bonus=support,
            extra_legs={"community": support_leg} if support_leg else None,
            top_k=top_k,
        )
        span.set_attribute("communities", len(communities))

    retrieval.strategy = "global_search"
    retrieval.communities = [c for _, c in key_points]
    retrieval.steps = [
        f"MAP sobre {len(key_points)} community reports (nivel {level})",
        "REDUCE: text units de suporte fundidos com a perna hibrida",
        *retrieval.steps,
    ]
    return retrieval


# ── DRIFT Search ──────────────────────────────────────────────────────────

def drift_search(index: Index, query: str, settings: Settings, *, top_k: int | None = None) -> Retrieval:
    """
    DRIFT = primer global -> perguntas de acompanhamento -> buscas locais -> reduce.

    E' a estrategia certa para perguntas que sao locais *e* tematicas ao mesmo
    tempo ("como o padrao X se compara ao Y no aspecto Z"), porque o primer
    descobre em qual regiao do grafo procurar antes de gastar saltos.
    """
    cfg = settings.agent
    top_k = top_k or settings.retrieval.top_k
    tracer = get_tracer()
    with tracer.span("retrieval.drift", depth=cfg.drift_depth, breadth=cfg.drift_breadth) as span:
        primer = global_search(index, query, settings, top_k=settings.retrieval.top_k)
        legs: dict[str, list[Scored]] = {"drift-primer": primer.context}
        followups: list[str] = []
        frontier = _followup_questions(index, query, primer, breadth=cfg.drift_breadth)

        for depth in range(cfg.drift_depth):
            next_frontier: list[str] = []
            for question in frontier:
                followups.append(question)
                local = local_search(index, question, settings, top_k=settings.retrieval.top_k)
                legs[f"drift-d{depth}-{len(followups)}"] = local.context
                if depth + 1 < cfg.drift_depth:
                    next_frontier.extend(
                        _followup_questions(index, question, local, breadth=1)
                    )
            frontier = next_frontier[: cfg.drift_breadth]
            if not frontier:
                break

        # A pergunta original pesa o dobro de cada follow-up.
        weights = {name: (2.0 if name == "drift-primer" else 1.0) for name in legs}
        fused = fuse(legs, method="rrf", k=settings.retrieval.rrf_k, weights=weights)
        reranked = _reranker.rerank(query, fused, index.chunks, top_n=settings.retrieval.rerank_top_n)
        selected = mmr(reranked, index.dense, top_k=top_k, lambda_=settings.retrieval.mmr_lambda)
        span.set_attribute("followups", len(followups))

    return Retrieval(
        "drift_search",
        context=selected,
        communities=primer.communities,
        subqueries=followups,
        expanded_entities=sorted({e for e in primer.expanded_entities}),
        steps=[
            f"primer global sobre {len(primer.communities)} comunidades",
            f"{len(followups)} perguntas de acompanhamento em {cfg.drift_depth} niveis",
            "fusao RRF de todas as rodadas + rerank + MMR",
        ],
    )


def _followup_questions(index: Index, query: str, retrieval: Retrieval, *, breadth: int) -> list[str]:
    """
    Gera perguntas de acompanhamento. Com LLM, pede-as explicitamente; sem LLM,
    deriva das entidades mais centrais do contexto recuperado que ainda nao
    aparecem na pergunta — que e' o que o LLM faria de qualquer forma.
    """
    if index.llm.name != "offline":
        payload = index.llm.structured(
            "Gere ate {n} perguntas de acompanhamento que ajudem a responder a pergunta "
            'original, usando o contexto. Responda em JSON: {{"questions": ["...", "..."]}}\n\n'
            "CONTEXTO:\n{context}\n\nPERGUNTA: {query}".format(
                n=breadth,
                context="\n".join(index.chunks[c.chunk_id].text for c in retrieval.context[:3] if c.chunk_id in index.chunks),
                query=query,
            )
        )
        if isinstance(payload, dict) and payload.get("questions"):
            return [str(q) for q in payload["questions"]][:breadth]

    asked = set(tokenize(query))
    ranked: list[tuple[float, str]] = []
    seen: set[str] = set()
    for scored in retrieval.context[:5]:
        chunk = index.chunks.get(scored.chunk_id)
        if not chunk:
            continue
        for name in chunk.entities:
            if name in seen or set(tokenize(name)) & asked:
                continue
            seen.add(name)
            ranked.append((index.graph.degree(name), name))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [f"Qual a relacao entre {name} e o tema da pergunta: {query}" for _, name in ranked[:breadth]]


# ── LazyGraphRAG ──────────────────────────────────────────────────────────

def lazy_graphrag(index: Index, query: str, settings: Settings, *, top_k: int | None = None) -> Retrieval:
    """
    Adia todo o trabalho caro para o momento da consulta.

    Nao le nenhum Community Report (portanto nao paga a indexacao que os
    gera). O fluxo e': candidatos hibridos baratos -> entidades presentes no
    topo -> BFS de poucos saltos -> teste de relevancia nos chunks alcancados
    -> ranqueamento final. Trocamos custo de indexacao por custo de consulta.
    """
    top_k = top_k or settings.retrieval.top_k
    tracer = get_tracer()
    with tracer.span("retrieval.lazy_graphrag") as span:
        seed_pool = index.bm25.search(query, top_n=10)
        dense_pool = index.dense.search(query, top_n=10)
        primer = fuse({"bm25": seed_pool, "dense": dense_pool}, method="rrf", k=settings.retrieval.rrf_k)

        seeds: list[str] = list(index.graph.link_entities(query))
        for scored in primer[:5]:
            chunk = index.chunks.get(scored.chunk_id)
            if chunk:
                seeds.extend(chunk.entities[:3])
        seeds = list(dict.fromkeys(seeds))

        reached = index.graph.bfs(seeds, hops=settings.graph.max_hops)
        # decaimento por salto: o chunk de um vizinho a 2 saltos vale menos
        graph_bonus: dict[str, float] = {}
        for name, distance in reached.items():
            entity = index.graph.entities.get(name)
            if not entity:
                continue
            weight = 1.0 / (1.0 + distance)
            for chunk_id in entity.chunks:
                graph_bonus[chunk_id] = max(graph_bonus.get(chunk_id, 0.0), weight)

        graph_leg = [
            Scored(chunk_id, weight, {"leg": "lazy-bfs"})
            for chunk_id, weight in sorted(graph_bonus.items(), key=lambda kv: -kv[1])
        ][: settings.retrieval.candidates_per_leg]

        retrieval = hybrid_search(
            index,
            query,
            settings,
            graph_bonus=graph_bonus,
            extra_legs={"lazy-bfs": graph_leg} if graph_leg else None,
            query_entities=list(reached),
            top_k=top_k,
        )
        span.set_attribute("seeds", len(seeds))
        span.set_attribute("reached", len(reached))

    retrieval.strategy = "lazy_graphrag"
    retrieval.seed_entities = seeds
    retrieval.expanded_entities = sorted(reached)
    retrieval.steps = [
        f"{len(seeds)} sementes (consulta + topo hibrido)",
        f"BFS {settings.graph.max_hops} saltos alcancou {len(reached)} entidades",
        "sem community reports — custo de indexacao zero",
        *retrieval.steps,
    ]
    return retrieval


STRATEGIES = {
    "bm25": bm25_search,
    "vector": vector_search,
    "hybrid": hybrid_search,
    "local_search": local_search,
    "global_search": global_search,
    "drift_search": drift_search,
    "lazy_graphrag": lazy_graphrag,
}


def run_strategy(name: str, index: Index, query: str, settings: Settings) -> Retrieval:
    if name not in STRATEGIES:
        raise KeyError(f"estrategia desconhecida: {name} (disponiveis: {', '.join(STRATEGIES)})")
    return STRATEGIES[name](index, query, settings)
