"""
Camada de recuperacao: indices lexical e denso, fusao e reranqueamento.

O desenho segue a pratica consolidada em 2026 para busca hibrida:

    query -> [BM25 top-N | denso top-N] -> fusao (RRF ou combinacao convexa)
          -> reranker sobre top-n -> MMR (diversidade) -> top-k

Duas fusoes estao implementadas porque elas erram de formas diferentes: RRF
opera sobre *posicoes* e e' imune a escalas incomparaveis; a combinacao convexa
opera sobre *scores* normalizados e preserva a margem entre o 1o e o 2o
colocado — melhor quando uma das pernas e' claramente superior na consulta.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .embeddings import Embedder, cosine
from .text import Chunk, tokenize


@dataclass
class Scored:
    """Resultado de recuperacao com proveniencia — o `provenance` alimenta a citacao."""

    chunk_id: str
    score: float
    provenance: dict = field(default_factory=dict)

    def with_score(self, score: float, **extra) -> "Scored":
        merged = {**self.provenance, **extra}
        return Scored(self.chunk_id, score, merged)


# ── Indice lexical ────────────────────────────────────────────────────────

class BM25Index:
    """Okapi BM25 sobre os text units. Referencia lexical do sistema."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: dict[str, int] = {}
        self.avg_len: float = 0.0
        self.term_freq: dict[str, dict[str, int]] = {}
        self.idf: dict[str, float] = {}

    def build(self, chunks: Sequence[Chunk]) -> "BM25Index":
        self.doc_ids = [c.chunk_id for c in chunks]
        postings: dict[str, dict[str, int]] = {}
        for chunk in chunks:
            self.doc_len[chunk.chunk_id] = len(chunk.tokens)
            for token in chunk.tokens:
                postings.setdefault(token, {}).setdefault(chunk.chunk_id, 0)
                postings[token][chunk.chunk_id] += 1
        self.term_freq = postings
        n_docs = max(len(chunks), 1)
        self.avg_len = sum(self.doc_len.values()) / n_docs
        # IDF com suavizacao de Robertson (a variante que nao vira negativa)
        self.idf = {
            term: math.log(1 + (n_docs - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in postings.items()
        }
        return self

    def search(self, query: str, top_n: int = 30) -> list[Scored]:
        scores: dict[str, float] = {}
        for term in tokenize(query):
            postings = self.term_freq.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for chunk_id, freq in postings.items():
                length = self.doc_len[chunk_id]
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * freq * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        return [Scored(cid, score, {"leg": "bm25"}) for cid, score in ranked]


# ── Indice denso ──────────────────────────────────────────────────────────

class DenseIndex:
    """Busca vetorial exaustiva. Em producao, trocar por HNSW/IVF sem mudar a interface."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.vectors: dict[str, list[float]] = {}

    def build(self, chunks: Sequence[Chunk]) -> "DenseIndex":
        self.embedder.fit([c.tokens for c in chunks])
        self.vectors = {c.chunk_id: self.embedder.encode(c.text) for c in chunks}
        return self

    def search(self, query: str, top_n: int = 30) -> list[Scored]:
        query_vector = self.embedder.encode(query)
        scores = ((cid, cosine(query_vector, vec)) for cid, vec in self.vectors.items())
        ranked = sorted(scores, key=lambda kv: (-kv[1], kv[0]))[:top_n]
        return [Scored(cid, score, {"leg": "dense"}) for cid, score in ranked if score > 0]

    def similarity(self, a: str, b: str) -> float:
        if a not in self.vectors or b not in self.vectors:
            return 0.0
        return cosine(self.vectors[a], self.vectors[b])


# ── Fusao ─────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    legs: dict[str, list[Scored]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[Scored]:
    """
    RRF ponderada: score = sum_l w_l / (k + rank_l).

    O peso importa quando as pernas nao sao pares. No DRIFT, por exemplo, a
    consulta original vale mais que cada pergunta de acompanhamento: sem peso,
    tres follow-ups genericos vencem por maioria a evidencia que responde a
    pergunta de fato.
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    provenance: dict[str, dict] = {}
    for leg_name, results in legs.items():
        weight = weights.get(leg_name, 1.0)
        for rank, item in enumerate(results, start=1):
            fused[item.chunk_id] = fused.get(item.chunk_id, 0.0) + weight / (k + rank)
            entry = provenance.setdefault(item.chunk_id, {"legs": {}})
            entry["legs"][leg_name] = rank
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Scored(cid, score, provenance[cid]) for cid, score in ranked]


def _min_max(results: Sequence[Scored]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r.score for r in results]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    return {r.chunk_id: (r.score - lo) / span for r in results}


def convex_combination(sparse: list[Scored], dense: list[Scored], alpha: float = 0.5) -> list[Scored]:
    """score = alpha * denso_norm + (1 - alpha) * sparse_norm, com min-max por perna."""
    sparse_norm, dense_norm = _min_max(sparse), _min_max(dense)
    fused: dict[str, float] = {}
    for cid in set(sparse_norm) | set(dense_norm):
        fused[cid] = alpha * dense_norm.get(cid, 0.0) + (1 - alpha) * sparse_norm.get(cid, 0.0)
    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        Scored(cid, score, {"legs": {"bm25": sparse_norm.get(cid), "dense": dense_norm.get(cid)}})
        for cid, score in ranked
    ]


def fuse(
    legs: dict[str, list[Scored]],
    *,
    method: str = "rrf",
    k: int = 60,
    alpha: float = 0.5,
    weights: dict[str, float] | None = None,
) -> list[Scored]:
    if method == "convex":
        return convex_combination(legs.get("bm25", []), legs.get("dense", []), alpha=alpha)
    return reciprocal_rank_fusion(legs, k=k, weights=weights)


# ── Reranqueamento ────────────────────────────────────────────────────────

class FeatureReranker:
    """
    Stand-in deterministico de um cross-encoder (ex.: bge-reranker-v2-m3).

    Combina sinais que um cross-encoder aprende implicitamente: cobertura dos
    termos da consulta, proximidade minima entre eles no texto, casamento de
    entidades e um bonus de grafo (distancia ate as entidades-semente). Deixar
    esses sinais explicitos torna o ranqueamento inspecionavel — cada ponto do
    score tem um nome no `provenance`.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "fusion": 1.0,
            "coverage": 1.4,
            "proximity": 0.6,
            "entity": 0.8,
            "graph": 0.7,
        }

    def score(
        self,
        query: str,
        candidate: Scored,
        chunk: Chunk,
        *,
        fused_norm: float,
        graph_bonus: float = 0.0,
        query_entities: Iterable[str] = (),
    ) -> tuple[float, dict]:
        query_terms = list(dict.fromkeys(tokenize(query)))
        chunk_tokens = chunk.tokens
        token_set = set(chunk_tokens)
        coverage = sum(1 for t in query_terms if t in token_set) / (len(query_terms) or 1)

        positions = [i for i, t in enumerate(chunk_tokens) if t in query_terms]
        if len(positions) > 1:
            window = min(b - a for a, b in zip(positions, positions[1:]))
            proximity = 1.0 / (1.0 + window)
        else:
            proximity = 0.25 if positions else 0.0

        entities = {e.lower() for e in chunk.entities}
        wanted = {e.lower() for e in query_entities}
        entity_hit = len(entities & wanted) / (len(wanted) or 1) if wanted else 0.0

        features = {
            "fusion": fused_norm,
            "coverage": coverage,
            "proximity": proximity,
            "entity": entity_hit,
            "graph": graph_bonus,
        }
        total = sum(self.weights[name] * value for name, value in features.items())
        return total, features

    def rerank(
        self,
        query: str,
        candidates: Sequence[Scored],
        chunks: dict[str, Chunk],
        *,
        top_n: int = 20,
        graph_bonus: dict[str, float] | None = None,
        query_entities: Iterable[str] = (),
    ) -> list[Scored]:
        head = list(candidates[:top_n])
        norm = _min_max(head)
        graph_bonus = graph_bonus or {}
        rescored: list[Scored] = []
        for candidate in head:
            chunk = chunks.get(candidate.chunk_id)
            if chunk is None:
                continue
            total, features = self.score(
                query,
                candidate,
                chunk,
                fused_norm=norm.get(candidate.chunk_id, 0.0),
                graph_bonus=graph_bonus.get(candidate.chunk_id, 0.0),
                query_entities=query_entities,
            )
            rescored.append(candidate.with_score(total, rerank=features))
        rescored.sort(key=lambda s: (-s.score, s.chunk_id))
        return rescored + list(candidates[top_n:])


def mmr(
    candidates: Sequence[Scored],
    dense: DenseIndex,
    *,
    top_k: int = 6,
    lambda_: float = 0.7,
) -> list[Scored]:
    """
    Maximal Marginal Relevance: penaliza candidatos redundantes.

    Em GraphRAG isso importa mais do que em RAG plano — a expansao por grafo
    tende a trazer varios chunks quase identicos sobre a mesma entidade, e o
    orcamento de contexto se esgota antes de cobrir a pergunta inteira.
    """
    if not candidates:
        return []
    pool = list(candidates)
    norm = _min_max(pool)
    selected: list[Scored] = [pool.pop(0)]
    while pool and len(selected) < top_k:
        best_idx, best_value = 0, float("-inf")
        for idx, candidate in enumerate(pool):
            redundancy = max(
                (dense.similarity(candidate.chunk_id, chosen.chunk_id) for chosen in selected),
                default=0.0,
            )
            value = lambda_ * norm.get(candidate.chunk_id, 0.0) - (1 - lambda_) * redundancy
            if value > best_value:
                best_idx, best_value = idx, value
        selected.append(pool.pop(best_idx))
    return selected
