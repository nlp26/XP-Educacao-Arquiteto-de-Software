"""
Pipeline de ingestao: corpus -> chunks -> indices -> grafo -> comunidades.

O `Index` resultante e' o objeto que todas as estrategias de consulta
compartilham. Construir uma vez e reutilizar e' proposital: o custo de
indexacao (extracao + relatorios) e' o que separa GraphRAG de LazyGraphRAG, e
so fica visivel quando ele e' medido separadamente do custo de consulta.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .communities import build_reports
from .config import Settings
from .embeddings import build_embedder
from .extraction import EntityExtractor
from .graph import KnowledgeGraph
from .llm import LLM, OfflineLLM, build_llm
from .observability import Tracer, get_tracer
from .retrievers import BM25Index, DenseIndex
from .text import Chunk, chunk_document


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class Index:
    settings: Settings
    documents: list[Document] = field(default_factory=list)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    bm25: BM25Index | None = None
    dense: DenseIndex | None = None
    graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    llm: LLM = field(default_factory=OfflineLLM)
    community_index: BM25Index | None = None
    community_by_id: dict = field(default_factory=dict)
    index_cost: dict[str, float] = field(default_factory=dict)

    @property
    def chunk_list(self) -> list[Chunk]:
        return list(self.chunks.values())

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "graph": self.graph.stats(),
            "index_cost": self.index_cost,
        }


def load_corpus(corpus_dir: str) -> list[Document]:
    """Le .md/.txt. A primeira linha `# Titulo` vira o titulo do documento."""
    documents: list[Document] = []
    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith((".md", ".txt")):
            continue
        path = os.path.join(corpus_dir, filename)
        with open(path, encoding="utf-8") as handle:
            raw = handle.read().strip()
        lines = raw.splitlines()
        if lines and lines[0].startswith("#"):
            title = lines[0].lstrip("# ").strip()
            body = "\n".join(lines[1:]).strip()
        else:
            title, body = os.path.splitext(filename)[0], raw
        documents.append(Document(doc_id=os.path.splitext(filename)[0], title=title, text=body))
    return documents


def build_index(settings: Settings, *, corpus_dir: str | None = None, tracer: Tracer | None = None) -> Index:
    tracer = tracer or get_tracer()
    corpus_dir = corpus_dir or settings.corpus_dir
    llm = build_llm(settings.llm_backend, settings.llm_model)
    use_llm = llm.name != "offline"

    with tracer.span("index.build", backend=llm.name, corpus=corpus_dir) as root:
        tracer.reset_counters(prefix="llm.")

        with tracer.span("index.load_and_chunk"):
            documents = load_corpus(corpus_dir)
            chunks: dict[str, Chunk] = {}
            for document in documents:
                for chunk in chunk_document(
                    document.doc_id,
                    document.title,
                    document.text,
                    target_tokens=settings.chunking.target_tokens,
                    overlap_sentences=settings.chunking.overlap_sentences,
                ):
                    chunks[chunk.chunk_id] = chunk

        with tracer.span("index.lexical"):
            bm25 = BM25Index(settings.retrieval.bm25_k1, settings.retrieval.bm25_b).build(list(chunks.values()))

        with tracer.span("index.dense", backend=settings.embedding_backend):
            embedder = build_embedder(settings.embedding_backend, settings.embedding_dim)
            dense = DenseIndex(embedder).build(list(chunks.values()))

        with tracer.span("index.extraction", llm=use_llm) as span:
            extractor = EntityExtractor(llm=llm, use_llm=use_llm)
            extractor.fit(list(chunks.values()), titles=[d.title for d in documents])
            graph = KnowledgeGraph()
            for chunk in chunks.values():
                entities, relations = extractor.extract(chunk)
                chunk.entities = [e.name for e in entities]
                for entity in entities:
                    entity.chunks.add(chunk.chunk_id)
                    graph.add_entity(entity)
                for relation in relations:
                    graph.add_relation(relation)
            merged = graph.resolve_entities()
            graph.prune(settings.graph.min_edge_weight)
            span.set_attribute("merged_entities", len(merged))
            span.set_attribute("entities", len(graph.entities))
            span.set_attribute("relations", len(graph.relations))

        with tracer.span("index.communities"):
            graph.detect_communities(levels=settings.graph.community_levels)

        build_reports(graph, chunks, llm, use_llm=use_llm)

        # Indice lexical sobre os relatorios: e' o que o Global Search consulta.
        community_chunks: list[Chunk] = []
        community_by_id: dict[str, object] = {}
        for level, communities in graph.communities.items():
            for community in communities:
                pseudo = Chunk(
                    chunk_id=community.community_id,
                    doc_id=f"community-l{level}",
                    title=community.title,
                    text=community.report_text or community.title,
                    position=level,
                )
                community_chunks.append(pseudo)
                community_by_id[community.community_id] = community
        community_index = BM25Index().build(community_chunks) if community_chunks else None

        index_cost = {
            "llm_calls_indexing": tracer.snapshot_counters().get("llm.calls", 0.0),
            "chunks": float(len(chunks)),
            "community_reports": tracer.snapshot_counters().get("index.reports", 0.0),
        }
        root.set_attribute("chunks", len(chunks))
        root.set_attribute("llm_calls", index_cost["llm_calls_indexing"])

    return Index(
        settings=settings,
        documents=documents,
        chunks=chunks,
        bm25=bm25,
        dense=dense,
        graph=graph,
        llm=llm,
        community_index=community_index,
        community_by_id=community_by_id,
        index_cost=index_cost,
    )
