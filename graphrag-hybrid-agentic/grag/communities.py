"""
Community Reports — o artefato que faz o Global Search funcionar.

Cada comunidade do grafo vira um resumo textual indexavel. Uma pergunta
tematica ("quais os temas centrais do corpus?") nao casa lexicalmente com
nenhum chunk isolado, mas casa com o *relatorio* que descreve o subgrafo.

Este e' tambem o ponto de custo do GraphRAG classico: uma chamada de LLM por
comunidade, em tempo de indexacao. O contador `llm.calls` do tracer separa esse
custo do custo de consulta — e' o que a tabela de avaliacao usa para contrastar
GraphRAG com LazyGraphRAG, que simplesmente nao executa esta etapa.
"""
from __future__ import annotations

from typing import Sequence

from .graph import Community, KnowledgeGraph
from .llm import LLM
from .observability import get_tracer
from .text import Chunk, sentences

REPORT_SYSTEM = "Voce resume subgrafos de conhecimento de forma factual e concisa."
REPORT_TEMPLATE = """Escreva um relatorio de comunidade.

Entidades: {entities}
Relacoes: {relations}

CONTEXTO:
{context}

Responda com JSON: {{"title": "titulo curto", "summary": "2 frases", "findings": ["achado 1", "achado 2", "achado 3"]}}
"""


def build_reports(
    graph: KnowledgeGraph,
    chunks: dict[str, Chunk],
    llm: LLM,
    *,
    use_llm: bool = False,
    max_context_chunks: int = 4,
) -> None:
    """Preenche title/summary/findings de cada comunidade, em todos os niveis."""
    tracer = get_tracer()
    with tracer.span("index.community_reports") as span:
        total = 0
        for level, communities in graph.communities.items():
            for community in communities:
                _fill_report(graph, chunks, community, llm, use_llm, max_context_chunks)
                total += 1
        span.set_attribute("reports", total)
        tracer.incr("index.reports", total)


def _fill_report(
    graph: KnowledgeGraph,
    chunks: dict[str, Chunk],
    community: Community,
    llm: LLM,
    use_llm: bool,
    max_context_chunks: int,
) -> None:
    ranked_members = sorted(community.members, key=lambda m: (-graph.degree(m), m))
    relations = [
        rel
        for key, rel in graph.relations.items()
        if key[0] in set(community.members) and key[1] in set(community.members)
    ]
    relations.sort(key=lambda r: -r.weight)
    context_ids = community.chunk_ids[:max_context_chunks]
    context = "\n".join(chunks[cid].text for cid in context_ids if cid in chunks)

    if use_llm:
        payload = llm.structured(
            REPORT_TEMPLATE.format(
                entities=", ".join(ranked_members[:12]),
                relations="; ".join(f"{r.source} --{r.predicate}--> {r.target}" for r in relations[:12]),
                context=context,
            ),
            system=REPORT_SYSTEM,
        )
        if isinstance(payload, dict) and payload.get("summary"):
            community.title = str(payload.get("title") or _fallback_title(ranked_members))
            community.summary = str(payload["summary"])
            community.findings = [str(f) for f in payload.get("findings", [])][:5]
            return

    # Fallback extrativo: titulo pelos hubs, resumo pelas frases mais densas em
    # entidades, achados pelas arestas de maior peso (a "espinha" do subgrafo).
    community.title = _fallback_title(ranked_members)
    community.summary = _extractive_summary(context, ranked_members)
    community.findings = [
        f"{rel.source} {rel.predicate} {rel.target}" for rel in relations[:5]
    ]


def _fallback_title(ranked_members: Sequence[str]) -> str:
    head = [m for m in ranked_members[:3]]
    return " / ".join(head) if head else "Comunidade"


def _extractive_summary(context: str, members: Sequence[str], max_sentences: int = 2) -> str:
    member_set = {m.lower() for m in members}
    scored: list[tuple[float, str]] = []
    for sentence in sentences(context):
        lowered = sentence.lower()
        hits = sum(1 for m in member_set if m in lowered)
        if hits:
            scored.append((hits / (len(sentence.split()) ** 0.5), sentence))
    scored.sort(key=lambda item: -item[0])
    return " ".join(s for _, s in scored[:max_sentences])
