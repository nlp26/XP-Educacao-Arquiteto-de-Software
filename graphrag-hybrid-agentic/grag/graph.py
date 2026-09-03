"""
Grafo de conhecimento: estrutura, deteccao de comunidades e caminhada.

Implementacao propria (sem networkx) para manter o exemplo sem dependencias e,
principalmente, para deixar visivel *o que* cada algoritmo faz — sao tres:

* **Label Propagation** para comunidades. O GraphRAG de referencia usa Leiden;
  LP e' o primo deterministico e O(E) dele. Com desempate por ordem lexical o
  resultado e' reproduzivel, que e' o requisito aqui. Ver ADR-004.
* **Personalized PageRank** para expansao no Local Search. Melhor que BFS puro
  porque pondera por importancia estrutural: um vizinho hub generico recebe
  menos massa que um vizinho especifico fortemente ligado a semente.
* **BFS limitada** para o LazyGraphRAG, onde a exploracao precisa ser barata e
  interrompivel.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .extraction import Entity, Relation
from .text import Chunk, normalize


@dataclass
class Community:
    community_id: str
    level: int
    members: list[str]
    chunk_ids: list[str] = field(default_factory=list)
    parent: str | None = None
    title: str = ""
    summary: str = ""
    findings: list[str] = field(default_factory=list)
    rank: float = 0.0

    @property
    def report_text(self) -> str:
        return " ".join([self.title, self.summary, *self.findings]).strip()

    def to_dict(self) -> dict:
        return {
            "community_id": self.community_id,
            "level": self.level,
            "members": self.members,
            "chunk_ids": self.chunk_ids,
            "parent": self.parent,
            "title": self.title,
            "summary": self.summary,
            "findings": self.findings,
            "rank": round(self.rank, 4),
        }


class KnowledgeGraph:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[tuple[str, str], Relation] = {}
        self.adjacency: dict[str, dict[str, float]] = {}
        self.communities: dict[int, list[Community]] = {}
        self._alias: dict[str, str] = {}  # forma normalizada -> nome canonico

    # ── construcao ────────────────────────────────────────────────────────
    def add_entity(self, entity: Entity) -> None:
        existing = self.entities.get(entity.name)
        if existing is None:
            self.entities[entity.name] = entity
            self.adjacency.setdefault(entity.name, {})
            self._alias[normalize(entity.name)] = entity.name
            return
        existing.chunks |= entity.chunks
        existing.mentions += entity.mentions
        if len(entity.description) > len(existing.description):
            existing.description = entity.description

    def add_relation(self, relation: Relation) -> None:
        if relation.source not in self.entities or relation.target not in self.entities:
            return
        key = relation.key()
        existing = self.relations.get(key)
        if existing is None:
            self.relations[key] = relation
        else:
            existing.weight += relation.weight
            existing.chunks |= relation.chunks
        weight = self.relations[key].weight
        self.adjacency.setdefault(relation.source, {})[relation.target] = weight
        self.adjacency.setdefault(relation.target, {})[relation.source] = weight

    # ── resolucao de entidades ────────────────────────────────────────────
    def resolve_entities(self) -> dict[str, str]:
        """
        Consolida entidades que denotam a mesma coisa. Sem esta etapa o grafo
        fragmenta: "DRIFT" e "DRIFT Search" viram nos distintos, o grau de cada
        um cai pela metade e o Local Search perde caminhos de dois saltos.

        Tres regras, todas conservadoras:
          1. *plural/flexao* — mesmo conjunto de radicais ("Community Report"
             e "Community Reports");
          2. *prefixo contido* — um nome e' prefixo do outro e so ocorre em
             chunks onde o mais longo tambem ocorre;
          3. *sigla* — as iniciais de um nome multi-palavra formam a sigla, e
             as duas aparecem no mesmo chunk.

        Retorna o mapa {nome_absorvido: nome_canonico}.
        """
        from .text import tokenize

        names = sorted(self.entities)
        token_key = {name: tuple(tokenize(name)) for name in names}
        merges: dict[str, str] = {}

        def canonical_of(a: str, b: str) -> tuple[str, str]:
            """Vence quem tem mais mencoes; empate desempata pelo nome mais curto."""
            ea, eb = self.entities[a], self.entities[b]
            if (ea.mentions, -len(a)) >= (eb.mentions, -len(b)):
                return a, b
            return b, a

        # 1. mesma assinatura de radicais
        by_key: dict[tuple[str, ...], list[str]] = {}
        for name in names:
            if token_key[name]:
                by_key.setdefault(token_key[name], []).append(name)
        for group in by_key.values():
            if len(group) < 2:
                continue
            keeper = max(group, key=lambda n: (self.entities[n].mentions, -len(n)))
            for other in group:
                if other != keeper:
                    merges[other] = keeper

        # 2. prefixo contido + 3. sigla
        for i, a in enumerate(names):
            if a in merges:
                continue
            for b in names[i + 1 :]:
                if b in merges or a in merges:
                    continue
                ka, kb = token_key[a], token_key[b]
                if not ka or not kb:
                    continue
                shorter, longer = (a, b) if len(ka) < len(kb) else (b, a)
                ks, kl = token_key[shorter], token_key[longer]
                same_prefix = len(ks) < len(kl) and kl[: len(ks)] == ks
                subset = self.entities[shorter].chunks <= self.entities[longer].chunks
                if same_prefix and subset:
                    merges[shorter] = longer
                    continue
                if _acronym_match(a, b) and (self.entities[a].chunks & self.entities[b].chunks):
                    keeper, absorbed = canonical_of(a, b)
                    merges[absorbed] = keeper

        # resolve cadeias (A -> B -> C)
        resolved: dict[str, str] = {}
        for source in merges:
            target = merges[source]
            seen = {source}
            while target in merges and target not in seen:
                seen.add(target)
                target = merges[target]
            if target != source:
                resolved[source] = target

        if resolved:
            self._apply_merges(resolved)
        return resolved

    def _apply_merges(self, merges: dict[str, str]) -> None:
        for absorbed, keeper in merges.items():
            source, target = self.entities.get(absorbed), self.entities.get(keeper)
            if source is None or target is None:
                continue
            target.chunks |= source.chunks
            target.mentions += source.mentions
            if len(source.description) > len(target.description):
                target.description = source.description
            del self.entities[absorbed]
            self._alias[normalize(absorbed)] = keeper

        rebuilt: dict[tuple[str, str], Relation] = {}
        for relation in self.relations.values():
            source = merges.get(relation.source, relation.source)
            target = merges.get(relation.target, relation.target)
            if source == target or source not in self.entities or target not in self.entities:
                continue
            key = (source, target) if source < target else (target, source)
            existing = rebuilt.get(key)
            if existing is None:
                relation.source, relation.target = key
                rebuilt[key] = relation
            else:
                existing.weight += relation.weight
                existing.chunks |= relation.chunks
        self.relations = rebuilt

        self.adjacency = {name: {} for name in self.entities}
        for (a, b), relation in self.relations.items():
            self.adjacency[a][b] = relation.weight
            self.adjacency[b][a] = relation.weight

    def prune(self, min_edge_weight: float = 1.0) -> None:
        """Remove arestas fracas (co-ocorrencia unica e ruidosa) e nos isolados sem mencoes."""
        for key, relation in list(self.relations.items()):
            if relation.weight < min_edge_weight:
                del self.relations[key]
                self.adjacency.get(relation.source, {}).pop(relation.target, None)
                self.adjacency.get(relation.target, {}).pop(relation.source, None)

    # ── consultas basicas ─────────────────────────────────────────────────
    def degree(self, name: str) -> float:
        return sum(self.adjacency.get(name, {}).values())

    def neighbors(self, name: str) -> dict[str, float]:
        return self.adjacency.get(name, {})

    def resolve(self, surface: str) -> str | None:
        return self._alias.get(normalize(surface))

    def link_entities(self, text: str, limit: int = 8) -> list[str]:
        """
        Entity linking por casamento de superficie sobre o texto da consulta.

        Casar a forma normalizada evita depender de NER na consulta — em
        perguntas curtas, o gazetteer do corpus e' um linker melhor que um
        modelo generico.
        """
        lowered = _scrub(text)
        hits: list[tuple[float, str]] = []
        for key, canonical in self._alias.items():
            key = _scrub(key)
            if len(key) < 3 or canonical not in self.entities:
                continue
            if f" {key} " in f" {lowered} ":
                # entidades mais longas sao mais especificas => pontuam mais
                hits.append((len(key) + self.entities[canonical].mentions * 0.1, canonical))
        hits.sort(key=lambda item: (-item[0], item[1]))
        # aliases distintos podem apontar para a mesma entidade canonica
        return list(dict.fromkeys(name for _, name in hits))[:limit]

    def chunks_for(self, names: Iterable[str]) -> list[str]:
        out: list[str] = []
        for name in names:
            entity = self.entities.get(name)
            if entity:
                out.extend(sorted(entity.chunks))
        return list(dict.fromkeys(out))

    # ── caminhada ─────────────────────────────────────────────────────────
    def personalized_pagerank(
        self,
        seeds: Sequence[str],
        *,
        alpha: float = 0.85,
        iterations: int = 30,
        tolerance: float = 1e-6,
    ) -> dict[str, float]:
        seeds = [s for s in seeds if s in self.entities]
        if not seeds:
            return {}
        restart = {name: 0.0 for name in self.entities}
        for seed in seeds:
            restart[seed] = 1.0 / len(seeds)
        rank = dict(restart)
        for _ in range(iterations):
            nxt = {name: (1 - alpha) * restart[name] for name in self.entities}
            for name, mass in rank.items():
                edges = self.adjacency.get(name, {})
                total = sum(edges.values())
                if not total:
                    nxt[name] += alpha * mass  # no isolado retem a propria massa
                    continue
                for neighbor, weight in edges.items():
                    nxt[neighbor] += alpha * mass * weight / total
            delta = sum(abs(nxt[n] - rank[n]) for n in rank)
            rank = nxt
            if delta < tolerance:
                break
        return dict(sorted(rank.items(), key=lambda kv: (-kv[1], kv[0])))

    def bfs(self, seeds: Sequence[str], hops: int = 2) -> dict[str, int]:
        """Retorna {entidade: distancia}. Barato e interrompivel — usado pelo LazyGraphRAG."""
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((s, 0) for s in seeds if s in self.entities)
        for seed, _ in list(queue):
            visited[seed] = 0
        while queue:
            name, distance = queue.popleft()
            if distance >= hops:
                continue
            for neighbor in sorted(self.adjacency.get(name, {})):
                if neighbor not in visited:
                    visited[neighbor] = distance + 1
                    queue.append((neighbor, distance + 1))
        return visited

    # ── comunidades ───────────────────────────────────────────────────────
    def detect_communities(self, levels: int = 2, max_iterations: int = 30) -> dict[int, list[Community]]:
        """
        Label Propagation deterministico, aplicado em cascata para obter hierarquia.

        Nivel 0 roda sobre o grafo de entidades; o nivel seguinte roda sobre o
        grafo *contraido* (uma comunidade vira um no), reproduzindo a hierarquia
        que o Leiden entrega nativamente.
        """
        nodes = sorted(self.entities)
        if not nodes:
            self.communities = {}
            return self.communities

        labels = {name: name for name in nodes}
        for _ in range(max_iterations):
            changed = False
            for name in nodes:
                edges = self.adjacency.get(name, {})
                if not edges:
                    continue
                votes: dict[str, float] = {}
                for neighbor, weight in edges.items():
                    label = labels[neighbor]
                    votes[label] = votes.get(label, 0.0) + weight
                best = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                if best != labels[name]:
                    labels[name] = best
                    changed = True
            if not changed:
                break

        grouped: dict[str, list[str]] = {}
        for name in nodes:
            grouped.setdefault(labels[name], []).append(name)

        result: dict[int, list[Community]] = {}
        level_zero = [
            Community(community_id=f"c0-{idx}", level=0, members=sorted(members))
            for idx, (_, members) in enumerate(sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0])))
        ]
        result[0] = level_zero

        # niveis superiores: agrega comunidades vizinhas ate `levels`
        current = level_zero
        for level in range(1, levels):
            current = self._merge_level(current, level)
            result[level] = current
            if len(current) <= 1:
                break

        for level_communities in result.values():
            for community in level_communities:
                community.chunk_ids = self.chunks_for(community.members)
                community.rank = sum(self.degree(m) for m in community.members)
        self.communities = result
        return result

    def _merge_level(self, children: list[Community], level: int) -> list[Community]:
        """Contrai o grafo: cada comunidade vira um no, arestas somam pesos entre membros."""
        owner = {member: child.community_id for child in children for member in child.members}
        super_edges: dict[tuple[str, str], float] = {}
        for (a, b), relation in self.relations.items():
            ca, cb = owner.get(a), owner.get(b)
            if ca is None or cb is None or ca == cb:
                continue
            key = (ca, cb) if ca < cb else (cb, ca)
            super_edges[key] = super_edges.get(key, 0.0) + relation.weight

        labels = {child.community_id: child.community_id for child in children}
        adjacency: dict[str, dict[str, float]] = {c.community_id: {} for c in children}
        for (ca, cb), weight in super_edges.items():
            adjacency[ca][cb] = weight
            adjacency[cb][ca] = weight

        for _ in range(20):
            changed = False
            for node in sorted(adjacency):
                votes: dict[str, float] = {}
                for neighbor, weight in adjacency[node].items():
                    votes[labels[neighbor]] = votes.get(labels[neighbor], 0.0) + weight
                if not votes:
                    continue
                best = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                if best != labels[node]:
                    labels[node] = best
                    changed = True
            if not changed:
                break

        grouped: dict[str, list[Community]] = {}
        for child in children:
            grouped.setdefault(labels[child.community_id], []).append(child)

        merged: list[Community] = []
        for idx, (_, group) in enumerate(sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
            members = sorted({m for child in group for m in child.members})
            community = Community(community_id=f"c{level}-{idx}", level=level, members=members)
            for child in group:
                child.parent = community.community_id
            merged.append(community)
        return merged

    # ── diagnostico ───────────────────────────────────────────────────────
    def stats(self) -> dict:
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "avg_degree": round(
                sum(len(v) for v in self.adjacency.values()) / (len(self.entities) or 1), 2
            ),
            "communities": {level: len(cs) for level, cs in self.communities.items()},
        }

    def to_dict(self) -> dict:
        return {
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations.values()],
            "communities": {
                str(level): [c.to_dict() for c in cs] for level, cs in self.communities.items()
            },
        }


def _acronym_match(a: str, b: str) -> bool:
    """"Maximal Marginal Relevance" <-> "MMR"."""
    for short, long_name in ((a, b), (b, a)):
        parts = long_name.split()
        if len(parts) < 2 or len(short.split()) != 1 or len(short) < 2:
            continue
        initials = "".join(word[0] for word in parts if word[0].isalpha())
        if normalize(short) == normalize(initials):
            return True
    return False


def _scrub(text: str) -> str:
    """Normaliza e remove pontuacao: "Event Sourcing?" deve casar com "event sourcing"."""
    return " ".join(re.sub(r"[^\w]+", " ", normalize(text)).split())
