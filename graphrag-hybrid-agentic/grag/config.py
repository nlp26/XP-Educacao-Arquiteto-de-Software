"""
Configuracao central do pipeline.

Tudo que e' ajustavel (parametros de BM25, fusao, grafo, agente) vive aqui,
para que um experimento seja reproduzido apenas trocando um `Settings`.
Variaveis de ambiente sobrescrevem os defaults — util em CI e em notebooks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


def _env(name: str, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class ChunkingSettings:
    target_tokens: int = 90          # tamanho alvo do text unit
    overlap_sentences: int = 1       # janela deslizante em frases, nao em tokens


@dataclass
class RetrievalSettings:
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    candidates_per_leg: int = 30     # top-N de cada perna (sparse / dense)
    rrf_k: int = 60                  # constante da Reciprocal Rank Fusion
    fusion: str = "rrf"              # "rrf" | "convex"
    convex_alpha: float = 0.5        # peso do denso na combinacao convexa
    rerank_top_n: int = 20           # profundidade do reranker
    mmr_lambda: float = 0.85         # 1.0 = so relevancia, 0.0 = so diversidade (ver README, varredura)
    top_k: int = 6                   # contexto final entregue ao gerador


@dataclass
class GraphSettings:
    min_edge_weight: int = 1
    ppr_alpha: float = 0.85          # damping do Personalized PageRank
    ppr_iterations: int = 30
    max_hops: int = 2
    community_levels: int = 2        # hierarquia: 0 = fina, 1 = agregada
    max_community_reports: int = 8   # top-M relatorios no global search


@dataclass
class AgentSettings:
    max_iterations: int = 3          # ciclos plan -> act -> critique
    groundedness_threshold: float = 0.55
    coverage_threshold: float = 0.60
    max_subquestions: int = 4
    drift_depth: int = 2
    drift_breadth: int = 3


@dataclass
class Settings:
    corpus_dir: str = "corpus"
    llm_backend: str = field(default_factory=lambda: _env("GRAG_LLM", "offline"))
    llm_model: str = field(default_factory=lambda: _env("GRAG_MODEL", "qwen2.5:7b"))
    embedding_backend: str = field(default_factory=lambda: _env("GRAG_EMBEDDINGS", "hashing"))
    embedding_dim: int = 384
    seed: int = 42
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    graph: GraphSettings = field(default_factory=GraphSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT = Settings()
