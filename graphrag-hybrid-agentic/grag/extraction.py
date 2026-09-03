"""
Extracao de entidades e relacoes — a etapa que transforma texto em grafo.

Dois caminhos, mesma saida:

1. **LLM** (quando ha backend): prompt unico por chunk pedindo JSON com
   entidades tipadas e relacoes com predicado. E' o caminho do GraphRAG
   classico da Microsoft, e o que domina o custo de indexacao.
2. **Heuristico** (default offline): gazetteer construido em duas passadas
   sobre o corpus. A primeira coleta candidatos (siglas, sequencias
   capitalizadas, negrito markdown, titulos); a segunda so aceita um candidato
   como entidade se ele aparecer capitalizado *fora* de inicio de frase ou for
   sigla — isso elimina o falso positivo classico ("Quando", "Cada").

A qualidade do grafo depende mais dessa etapa do que de qualquer parametro de
recuperacao: uma aresta errada aqui propaga para o PPR, para as comunidades e
para o DRIFT.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .llm import LLM
from .text import Chunk, normalize, sentences, tokenize

_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6}(?:-[A-Z0-9]{1,4})?)\b")
_PROPER_RE = re.compile(r"\b([A-ZÀ-Ú][\wÀ-ú]+(?:[ -](?:de|da|do|of|the)?\s?[A-ZÀ-Ú][\wÀ-ú]+){0,3})\b")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")

_TYPE_HINTS = {
    "padrao": ("cqrs", "saga", "sidecar", "circuit", "outbox", "strangler", "mvc", "bff"),
    "tecnologia": ("kafka", "postgres", "redis", "neo4j", "qdrant", "opentelemetry", "langgraph", "ollama", "faiss", "elasticsearch"),
    "metrica": ("ndcg", "recall", "latencia", "mrr", "precision", "throughput"),
    "tecnica": ("rrf", "bm25", "hnsw", "mmr", "ppr", "leiden", "drift", "graphrag", "rag", "reranker", "embedding"),
}


@dataclass
class Entity:
    name: str
    type: str = "conceito"
    description: str = ""
    chunks: set[str] = field(default_factory=set)
    mentions: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "mentions": self.mentions,
            "chunks": sorted(self.chunks),
        }


@dataclass
class Relation:
    source: str
    target: str
    predicate: str = "relaciona-se com"
    weight: float = 1.0
    chunks: set[str] = field(default_factory=set)
    description: str = ""

    def key(self) -> tuple[str, str]:
        return tuple(sorted((self.source, self.target)))  # type: ignore[return-value]

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "predicate": self.predicate,
            "weight": self.weight,
            "chunks": sorted(self.chunks),
            "description": self.description,
        }


LLM_SYSTEM = (
    "Voce e' um extrator de grafos de conhecimento. Responda SOMENTE com JSON valido."
)
LLM_TEMPLATE = """Extraia entidades e relacoes do texto.

Formato exigido:
{{"entities": [{{"name": "...", "type": "conceito|padrao|tecnologia|tecnica|metrica", "description": "..."}}],
 "relations": [{{"source": "...", "target": "...", "predicate": "verbo curto", "description": "..."}}]}}

Use apenas nomes que aparecem no texto. Maximo 8 entidades e 8 relacoes.

CONTEXTO:
{text}
"""


def _infer_type(name: str) -> str:
    key = normalize(name)
    for type_name, hints in _TYPE_HINTS.items():
        if any(hint in key for hint in hints):
            return type_name
    if name.isupper() and len(name) <= 6:
        return "tecnica"
    return "conceito"


class EntityExtractor:
    """Extrator com gazetteer global e refinamento por chunk."""

    def __init__(self, llm: LLM | None = None, use_llm: bool = False) -> None:
        self.llm = llm
        self.use_llm = use_llm and llm is not None
        self.gazetteer: dict[str, str] = {}  # forma normalizada -> forma canonica

    # ── passada 1: descobrir o vocabulario de entidades do corpus ─────────
    def fit(self, chunks: Sequence[Chunk], titles: Sequence[str] = ()) -> "EntityExtractor":
        candidates: Counter[str] = Counter()
        confirmed: set[str] = set()

        for title in titles:
            confirmed.add(title.strip())

        for chunk in chunks:
            for bold in _BOLD_RE.findall(chunk.text):
                confirmed.add(bold.strip())
            for sentence in sentences(chunk.text):
                acronyms = _ACRONYM_RE.findall(sentence)
                confirmed.update(acronyms)
                first_word = sentence.split(" ", 1)[0].strip(".,:;")
                for match in _PROPER_RE.findall(sentence):
                    name = match.strip()
                    candidates[name] += 1
                    # capitalizado fora do inicio da frase => nome proprio de verdade
                    if not name.startswith(first_word) or sentence.find(name) > 0:
                        confirmed.add(name)

        gazetteer: dict[str, str] = {}
        for name in confirmed:
            clean = name.strip(" .,:;()").strip()
            if len(clean) < 2 or normalize(clean) in {"o", "a", "os", "as"}:
                continue
            if len(clean.split()) > 5:
                continue
            key = normalize(clean)
            # mantem a grafia mais frequente / mais curta como canonica
            current = gazetteer.get(key)
            if current is None or (candidates[clean] > candidates.get(current, 0)):
                gazetteer[key] = clean
        self.gazetteer = gazetteer
        return self

    # ── passada 2: anotar cada chunk ──────────────────────────────────────
    def extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        if self.use_llm:
            result = self._extract_with_llm(chunk)
            if result is not None:
                return result
        return self._extract_heuristic(chunk)

    def _extract_with_llm(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]] | None:
        assert self.llm is not None
        payload = self.llm.structured(LLM_TEMPLATE.format(text=chunk.text), system=LLM_SYSTEM)
        if not isinstance(payload, dict) or "entities" not in payload:
            return None
        entities: list[Entity] = []
        for item in payload.get("entities", []):
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            entities.append(
                Entity(
                    name=name,
                    type=str(item.get("type") or _infer_type(name)),
                    description=str(item.get("description", "")),
                    chunks={chunk.chunk_id},
                    mentions=1,
                )
            )
        known = {e.name for e in entities}
        relations: list[Relation] = []
        for item in payload.get("relations", []):
            source, target = str(item.get("source", "")).strip(), str(item.get("target", "")).strip()
            if source in known and target in known and source != target:
                relations.append(
                    Relation(
                        source=source,
                        target=target,
                        predicate=str(item.get("predicate", "relaciona-se com")),
                        chunks={chunk.chunk_id},
                        description=str(item.get("description", "")),
                    )
                )
        return entities, relations

    def _extract_heuristic(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        found: dict[str, Entity] = {}
        per_sentence: list[tuple[str, list[str]]] = []

        for sentence in sentences(chunk.text):
            lowered = normalize(sentence)
            present: list[str] = []
            for key, canonical in self.gazetteer.items():
                if re.search(rf"(?<![\w]){re.escape(key)}(?![\w])", lowered):
                    present.append(canonical)
                    entity = found.get(canonical)
                    if entity is None:
                        entity = Entity(
                            name=canonical,
                            type=_infer_type(canonical),
                            description=sentence.strip(),
                            chunks={chunk.chunk_id},
                        )
                        found[canonical] = entity
                    entity.mentions += 1
            per_sentence.append((sentence, sorted(set(present))))

        relations: dict[tuple[str, str], Relation] = {}
        for sentence, present in per_sentence:
            for i, source in enumerate(present):
                for target in present[i + 1 :]:
                    key = (source, target) if source < target else (target, source)
                    relation = relations.get(key)
                    if relation is None:
                        relation = Relation(
                            source=key[0],
                            target=key[1],
                            predicate=_predicate_between(sentence, key[0], key[1]),
                            chunks={chunk.chunk_id},
                            description=sentence.strip(),
                            weight=0.0,
                        )
                        relations[key] = relation
                    relation.weight += 1.0
                    relation.chunks.add(chunk.chunk_id)
        return list(found.values()), list(relations.values())


def _predicate_between(sentence: str, a: str, b: str) -> str:
    """Predicado = primeiro token de conteudo entre as duas mencoes."""
    lowered = normalize(sentence)
    ia, ib = lowered.find(normalize(a)), lowered.find(normalize(b))
    if ia == -1 or ib == -1:
        return "relaciona-se com"
    start, end = (ia + len(a), ib) if ia < ib else (ib + len(b), ia)
    middle = tokenize(sentence[start:end])
    return middle[0] if middle else "relaciona-se com"
