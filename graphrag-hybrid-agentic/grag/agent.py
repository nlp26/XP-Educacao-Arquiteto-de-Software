"""
Camada agentica: roteamento adaptativo, decomposicao, sintese e critica.

O ciclo e' `plan -> retrieve -> synthesize -> critique -> (replan)`, com no
maximo `agent.max_iterations` voltas. Tres decisoes de projeto merecem nota:

1. **Rotear antes de recuperar.** A estrategia mais cara nao pode ser paga em
   toda consulta. O classificador separa quatro tipos de pergunta (factual,
   relacional, tematica, comparativa) e cada tipo tem um caminho proprio.
2. **Criticar antes de responder.** O critico mede fundamentacao (a resposta
   esta nos trechos citados?) e cobertura (a pergunta foi respondida por
   inteiro?). Sao dois modos de falha diferentes e exigem correcoes diferentes.
3. **Escalar, nao repetir.** Reprovar nao dispara a mesma busca de novo: a
   proxima iteracao troca de estrategia, amplia o top-k e decompoe a pergunta.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Settings
from .observability import get_tracer
from .pipeline import Index
from .strategies import Retrieval, run_strategy
from .text import normalize, sentences, tokenize

# ── Classificacao ─────────────────────────────────────────────────────────

THEMATIC_MARKERS = (
    "quais sao os temas", "temas centrais", "visao geral", "panorama", "resumo do corpus",
    "principais topicos", "de modo geral", "o que o corpus cobre", "quais assuntos",
)
COMPARATIVE_MARKERS = (
    "compare", "comparacao", "diferenca", "diferencas", "versus", " vs ", "em vez de",
    "trade-off", "tradeoff", "quando usar", "melhor que", "vantagem sobre",
)
RELATIONAL_MARKERS = (
    "relacao", "relaciona", "depende", "impacta", "afeta", "como se conecta",
    "por que", "porque", "qual o papel", "influencia", "garante", "sustenta",
)

ROUTING = {
    "factual": "hybrid",
    "relacional": "local_search",
    "tematica": "global_search",
    "comparativa": "drift_search",
}

# Escalonamento usado quando o critico reprova a resposta.
ESCALATION = {
    "hybrid": "local_search",
    "local_search": "drift_search",
    "global_search": "drift_search",
    "drift_search": "drift_search",
    "lazy_graphrag": "local_search",
    "bm25": "hybrid",
    "vector": "hybrid",
}

CLASSIFIER_PROMPT = """Classifique a pergunta em uma categoria e proponha sub-perguntas se necessario.

Categorias:
- factual: fato pontual, uma unica evidencia basta
- relacional: exige ligar duas ou mais entidades (multi-hop)
- tematica: pede panorama/temas do corpus inteiro
- comparativa: contrasta duas alternativas

Responda em JSON: {{"kind": "...", "subquestions": ["...", "..."], "reason": "..."}}

PERGUNTA: {query}
"""


@dataclass
class Plan:
    kind: str
    strategy: str
    subqueries: list[str] = field(default_factory=list)
    reason: str = ""
    top_k: int = 6

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "strategy": self.strategy,
            "subqueries": self.subqueries,
            "reason": self.reason,
            "top_k": self.top_k,
        }


@dataclass
class Critique:
    groundedness: float
    coverage: float
    citations_valid: bool
    passed: bool
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "groundedness": round(self.groundedness, 3),
            "coverage": round(self.coverage, 3),
            "citations_valid": self.citations_valid,
            "passed": self.passed,
            "gaps": self.gaps,
        }


@dataclass
class AgentAnswer:
    query: str
    answer: str
    citations: list[dict]
    plan: Plan
    retrieval: Retrieval
    critique: Critique
    iterations: int
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": self.citations,
            "plan": self.plan.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "critique": self.critique.to_dict(),
            "iterations": self.iterations,
            "history": self.history,
        }


class AdaptiveRouter:
    """Classificador de complexidade — heuristico por padrao, LLM quando disponivel."""

    def __init__(self, index: Index, settings: Settings) -> None:
        self.index = index
        self.settings = settings

    def plan(self, query: str) -> Plan:
        with get_tracer().span("agent.plan") as span:
            plan = self._plan_with_llm(query) or self._plan_heuristic(query)
            span.set_attribute("kind", plan.kind)
            span.set_attribute("strategy", plan.strategy)
            return plan

    def _plan_with_llm(self, query: str) -> Plan | None:
        if self.index.llm.name == "offline":
            return None
        payload = self.index.llm.structured(CLASSIFIER_PROMPT.format(query=query))
        if not isinstance(payload, dict) or payload.get("kind") not in ROUTING:
            return None
        kind = str(payload["kind"])
        return Plan(
            kind=kind,
            strategy=ROUTING[kind],
            subqueries=[str(q) for q in payload.get("subquestions", [])][: self.settings.agent.max_subquestions],
            reason=str(payload.get("reason", "classificado pelo LLM")),
            top_k=self.settings.retrieval.top_k,
        )

    def _plan_heuristic(self, query: str) -> Plan:
        # Marcadores sao casados por palavra inteira: sem isso "dependencia"
        # dispara o marcador "depende" e a pergunta factual vira relacional.
        lowered = " " + " ".join(re.sub(r"[^\w]+", " ", normalize(query)).split()) + " "
        entities = self.index.graph.link_entities(query)

        def has(markers: tuple[str, ...]) -> bool:
            return any(f" {marker.strip()} " in lowered for marker in markers)

        if has(THEMATIC_MARKERS):
            kind, reason = "tematica", "marcadores de pergunta panoramica"
        elif has(COMPARATIVE_MARKERS):
            kind, reason = "comparativa", "marcadores de contraste entre alternativas"
        elif len(entities) >= 2 or has(RELATIONAL_MARKERS):
            kind, reason = "relacional", f"{len(entities)} entidades casadas no grafo"
        else:
            kind, reason = "factual", "pergunta pontual, sem sinal relacional"

        # Sem ancora no grafo, o Local Search nao tem de onde partir.
        strategy = ROUTING[kind]
        if strategy == "local_search" and not entities:
            strategy, reason = "hybrid", "sem entidade ancorada: degradando para hibrida"

        return Plan(
            kind=kind,
            strategy=strategy,
            subqueries=self.decompose(query, kind, entities),
            reason=reason,
            top_k=self.settings.retrieval.top_k,
        )

    def decompose(self, query: str, kind: str, entities: list[str]) -> list[str]:
        """Decomposicao de consulta: so vale a pena para comparativa e relacional."""
        if kind not in ("comparativa", "relacional") or len(entities) < 2:
            return []
        head = entities[: self.settings.agent.max_subquestions]
        subqueries = [f"O que e' {name} e qual o seu papel?" for name in head[:2]]
        if len(head) >= 2:
            subqueries.append(f"Qual a relacao entre {head[0]} e {head[1]}?")
        return subqueries


# ── Sintese ───────────────────────────────────────────────────────────────

ANSWER_SYSTEM = (
    "Responda apenas com base no contexto. Cite os trechos usando marcadores [n]. "
    "Se o contexto nao contiver a resposta, diga isso explicitamente."
)


def synthesize(index: Index, query: str, retrieval: Retrieval) -> tuple[str, list[dict]]:
    """Gera a resposta e liga cada frase ao trecho que a sustenta."""
    with get_tracer().span("agent.synthesize", chunks=len(retrieval.context)):
        citations: list[dict] = []
        blocks: list[str] = []
        for position, scored in enumerate(retrieval.context, start=1):
            chunk = index.chunks.get(scored.chunk_id)
            if not chunk:
                continue
            citations.append(
                {
                    "marker": position,
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "score": round(scored.score, 4),
                }
            )
            blocks.append(f"[{position}] ({chunk.title}) {chunk.text}")

        if not blocks:
            return "Nao encontrei evidencia no corpus para responder a essa pergunta.", []

        prompt = "CONTEXTO:\n" + "\n\n".join(blocks) + f"\n\nPERGUNTA: {query}"
        raw = index.llm.complete(prompt, system=ANSWER_SYSTEM)
        answer = _attach_markers(raw, index, retrieval, citations)
        used = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
        return answer, [c for c in citations if c["marker"] in used] or citations[:1]


def _attach_markers(raw: str, index: Index, retrieval: Retrieval, citations: list[dict]) -> str:
    """
    Se o gerador nao citou, ancoramos cada frase ao trecho mais proximo.

    Preferir marcacao automatica a exigir do modelo e' deliberado: a citacao
    passa a ser verificavel mesmo com um gerador extrativo, e o critico mede
    fundamentacao real em vez de obediencia de formato.
    """
    if re.search(r"\[\d+\]", raw):
        return raw.strip()

    marker_by_chunk = {c["chunk_id"]: c["marker"] for c in citations}
    annotated: list[str] = []
    for sentence in sentences(raw):
        terms = set(tokenize(sentence))
        if not terms:
            continue
        best_marker, best_overlap = None, 0.0
        for scored in retrieval.context:
            chunk = index.chunks.get(scored.chunk_id)
            if not chunk:
                continue
            overlap = len(terms & set(chunk.tokens)) / (len(terms) or 1)
            if overlap > best_overlap:
                best_marker, best_overlap = marker_by_chunk.get(chunk.chunk_id), overlap
        annotated.append(f"{sentence} [{best_marker}]" if best_marker else sentence)
    return " ".join(annotated).strip()


# ── Critica ───────────────────────────────────────────────────────────────

def critique(
    index: Index,
    query: str,
    answer: str,
    citations: list[dict],
    retrieval: Retrieval,
    settings: Settings,
) -> Critique:
    """
    Duas metricas independentes, porque sao dois modos de falha distintos.

    * fundamentacao baixa  -> o gerador extrapolou: reduzir contexto ruidoso
    * cobertura baixa      -> a recuperacao nao trouxe a evidencia: escalar
    """
    with get_tracer().span("agent.critique") as span:
        cited_tokens: set[str] = set()
        valid = True
        cited_ids = {c["chunk_id"] for c in citations}
        for chunk_id in cited_ids:
            chunk = index.chunks.get(chunk_id)
            if chunk is None:
                valid = False
                continue
            cited_tokens |= set(chunk.tokens)

        answer_tokens = [t for t in tokenize(answer)]
        grounded = (
            sum(1 for t in answer_tokens if t in cited_tokens) / len(answer_tokens)
            if answer_tokens
            else 0.0
        )

        # Um termo que nao existe em lugar nenhum do corpus nao pode ser
        # "coberto"; conta-lo como lacuna puniria a recuperacao por uma
        # limitacao do corpus e dispararia escalonamentos inuteis.
        vocabulary = set(index.bm25.term_freq) if index.bm25 else set()
        all_terms = list(dict.fromkeys(tokenize(query)))
        query_terms = [t for t in all_terms if t in vocabulary] or all_terms
        out_of_corpus = [t for t in all_terms if t not in vocabulary]
        context_tokens: set[str] = set()
        for scored in retrieval.context:
            chunk = index.chunks.get(scored.chunk_id)
            if chunk:
                context_tokens |= set(chunk.tokens)
        answered = set(answer_tokens)
        covered = [t for t in query_terms if t in answered or t in context_tokens]
        coverage = len(covered) / (len(query_terms) or 1)
        gaps = [t for t in query_terms if t not in covered]
        if out_of_corpus:
            span.set_attribute("out_of_corpus", ",".join(out_of_corpus[:5]))

        passed = (
            valid
            and grounded >= settings.agent.groundedness_threshold
            and coverage >= settings.agent.coverage_threshold
        )
        span.set_attribute("groundedness", round(grounded, 3))
        span.set_attribute("coverage", round(coverage, 3))
        span.set_attribute("passed", passed)
        return Critique(grounded, coverage, valid, passed, gaps)


# ── Ciclo agentico ────────────────────────────────────────────────────────

class GraphRAGAgent:
    def __init__(self, index: Index, settings: Settings | None = None) -> None:
        self.index = index
        self.settings = settings or index.settings
        self.router = AdaptiveRouter(index, self.settings)

    def answer(self, query: str, *, force_strategy: str | None = None) -> AgentAnswer:
        tracer = get_tracer()
        with tracer.span("agent.answer", query=query[:80]) as span:
            tracer.reset_counters(prefix="llm.")
            plan = self.router.plan(query)
            if force_strategy:
                plan.strategy = force_strategy
                plan.reason = f"estrategia forcada: {force_strategy}"

            history: list[dict] = []
            best: tuple[float, AgentAnswer] | None = None

            for iteration in range(1, self.settings.agent.max_iterations + 1):
                with tracer.span("agent.iteration", n=iteration, strategy=plan.strategy):
                    retrieval = self._retrieve(query, plan)
                    answer_text, citations = synthesize(self.index, query, retrieval)
                    verdict = critique(
                        self.index, query, answer_text, citations, retrieval, self.settings
                    )
                    history.append(
                        {
                            "iteration": iteration,
                            "strategy": plan.strategy,
                            "top_k": plan.top_k,
                            "chunks": retrieval.chunk_ids,
                            "critique": verdict.to_dict(),
                        }
                    )
                    candidate = AgentAnswer(
                        query=query,
                        answer=answer_text,
                        citations=citations,
                        plan=plan,
                        retrieval=retrieval,
                        critique=verdict,
                        iterations=iteration,
                        history=history,
                    )
                    quality = verdict.groundedness + verdict.coverage
                    if best is None or quality > best[0]:
                        best = (quality, candidate)
                    if verdict.passed:
                        span.set_attribute("iterations", iteration)
                        span.set_attribute("resolved", True)
                        return candidate
                    plan = self._replan(plan, verdict, query)

            span.set_attribute("iterations", self.settings.agent.max_iterations)
            span.set_attribute("resolved", False)
            assert best is not None
            return best[1]

    def _retrieve(self, query: str, plan: Plan) -> Retrieval:
        """Executa a estrategia; com sub-perguntas, funde as rodadas por RRF."""
        from .retrievers import fuse

        settings = self._settings_for(plan)
        main = run_strategy(plan.strategy, self.index, query, settings)
        if not plan.subqueries:
            return main

        legs = {"main": main.context}
        for position, subquery in enumerate(plan.subqueries):
            sub = run_strategy(plan.strategy, self.index, subquery, settings)
            legs[f"sub{position}"] = sub.context
        weights = {name: (2.0 if name == "main" else 1.0) for name in legs}
        fused = fuse(legs, method="rrf", k=settings.retrieval.rrf_k, weights=weights)[: plan.top_k]
        main.context = fused
        main.subqueries = plan.subqueries
        main.steps.append(f"decomposicao: {len(plan.subqueries)} sub-perguntas fundidas por RRF")
        return main

    def _settings_for(self, plan: Plan) -> Settings:
        """Clona as settings ajustando apenas o top-k desta iteracao."""
        import copy

        settings = copy.deepcopy(self.settings)
        settings.retrieval.top_k = plan.top_k
        return settings

    def _replan(self, plan: Plan, verdict: Critique, query: str) -> Plan:
        """Escalar: trocar de estrategia, ampliar o contexto e decompor."""
        strategy = ESCALATION.get(plan.strategy, "drift_search")
        subqueries = list(plan.subqueries)
        if verdict.coverage < self.settings.agent.coverage_threshold and verdict.gaps:
            for gap in verdict.gaps[:2]:
                candidate = f"O que o corpus diz sobre {gap}?"
                if candidate not in subqueries:
                    subqueries.append(candidate)
        get_tracer().event(
            "replan",
            from_strategy=plan.strategy,
            to_strategy=strategy,
            reason="cobertura" if verdict.coverage < self.settings.agent.coverage_threshold else "fundamentacao",
        )
        return Plan(
            kind=plan.kind,
            strategy=strategy,
            subqueries=subqueries[: self.settings.agent.max_subquestions],
            reason=f"escalonado apos reprovacao ({'cobertura' if verdict.coverage < self.settings.agent.coverage_threshold else 'fundamentacao'})",
            top_k=min(plan.top_k + 3, 12),
        )
