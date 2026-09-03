"""
Camada de embeddings com dois backends.

`HashingEmbedder` (default) e' deterministico e offline: projeta tokens
*e* n-gramas de caractere num espaco de dimensao fixa via hashing com sinal
(o "hashing trick" de Weinberger et al.), ponderados por IDF. Nao substitui um
encoder treinado, mas produz um sinal *semanticamente distinto* do BM25 — ele
casa variacoes morfologicas e composicoes que o lexical perde — o que e'
exatamente o que a fusao hibrida precisa para ter ganho real.

`OllamaEmbedder` chama um modelo de embedding local (ex.: nomic-embed-text)
por HTTP puro. Em producao, troque por um encoder denso de verdade; a
interface `Embedder` isola o resto do pipeline dessa decisao.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from typing import Protocol, Sequence

from .text import char_ngrams, tokenize

Vector = list[float]


class Embedder(Protocol):
    dim: int

    def fit(self, corpus_tokens: Sequence[Sequence[str]]) -> None: ...
    def encode(self, text: str) -> Vector: ...


def cosine(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))  # vetores ja normalizados em L2


def _hash_feature(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


class HashingEmbedder:
    """Bag-of-features com hashing assinado, ponderado por IDF e normalizado."""

    def __init__(self, dim: int = 384, ngram_weight: float = 0.35, ngram_size: int = 4) -> None:
        self.dim = dim
        self.ngram_weight = ngram_weight
        self.ngram_size = ngram_size
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0

    def fit(self, corpus_tokens: Sequence[Sequence[str]]) -> None:
        n_docs = max(len(corpus_tokens), 1)
        df: dict[str, int] = {}
        for tokens in corpus_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        self._idf = {t: math.log(1 + n_docs / (1 + c)) for t, c in df.items()}
        self._default_idf = math.log(1 + n_docs)

    def _weight(self, token: str) -> float:
        return self._idf.get(token, self._default_idf)

    def encode(self, text: str) -> Vector:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            weight = self._weight(token)
            idx, sign = _hash_feature(f"w::{token}", self.dim)
            vector[idx] += sign * weight
            sub_weight = self.ngram_weight * weight / math.sqrt(len(token) + 1)
            for gram in char_ngrams(token, self.ngram_size):
                g_idx, g_sign = _hash_feature(f"g::{gram}", self.dim)
                vector[g_idx] += g_sign * sub_weight
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OllamaEmbedder:
    """Encoder denso real via Ollama (`ollama pull nomic-embed-text`)."""

    def __init__(self, model: str = "nomic-embed-text", host: str = "http://localhost:11434", dim: int = 768) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.dim = dim

    def fit(self, corpus_tokens: Sequence[Sequence[str]]) -> None:  # noqa: D401 - sem estado
        return None

    def encode(self, text: str) -> Vector:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
        vector = data["embedding"]
        self.dim = len(vector)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def build_embedder(backend: str, dim: int, model: str = "nomic-embed-text") -> Embedder:
    if backend == "ollama":
        return OllamaEmbedder(model=model)
    return HashingEmbedder(dim=dim)
