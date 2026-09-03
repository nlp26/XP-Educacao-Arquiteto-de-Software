"""
Processamento textual: normalizacao, tokenizacao, stemming leve e chunking.

Optou-se por implementacao propria (stdlib) em vez de spaCy/NLTK por tres
motivos: (a) o exemplo precisa rodar sem downloads; (b) o corpus e' bilingue
pt/en e um stemmer sufixal cobre bem os dois; (c) manter a tokenizacao
explicita torna o BM25 auditavel — o mesmo tokenizador alimenta o indice
lexical, o embedder e o extrator de entidades.
"""
from __future__ import annotations

import re
import unicodedata

# Stopwords pt + en. Lista curta e' proposital: remover demais prejudica
# perguntas curtas ("o que e' CQRS?") mais do que ajuda o BM25.
STOPWORDS = {
    # portugues
    "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "do", "dos", "e",
    "em", "entre", "essa", "esse", "esta", "este", "eu", "foi", "for", "isso",
    "mais", "mas", "na", "nas", "no", "nos", "num", "numa", "o", "os", "ou",
    "para", "pela", "pelo", "por", "qual", "quais", "quando", "que", "quem",
    "se", "sem", "ser", "seu", "sua", "sao", "sobre", "tem", "um", "uma", "voce",
    # ingles
    "an", "and", "are", "at", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "was", "what", "which", "with",
}

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.]*", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?:])\s+(?=[A-ZÀ-Ú0-9])")

# Sufixos ordenados do mais longo para o mais curto (guloso).
_SUFFIXES = (
    "acoes", "amento", "mentos", "encia", "ancia", "idade", "ismo", "ista",
    "cao", "coes", "ies", "ing", "ers", "es", "as", "os", "s",
)


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize(text: str) -> str:
    return strip_accents(text).lower()


def stem(token: str) -> str:
    """Stemmer sufixal conservador: so corta se sobrar um radical util (>=4)."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str, *, keep_stopwords: bool = False, apply_stem: bool = True) -> list[str]:
    tokens = []
    for raw in _TOKEN_RE.findall(normalize(text)):
        token = raw.strip(".-")
        if not token or len(token) < 2:
            continue
        if not keep_stopwords and token in STOPWORDS:
            continue
        tokens.append(stem(token) if apply_stem else token)
    return tokens


def sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text.replace("\n", " ")) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def char_ngrams(token: str, n: int = 4) -> list[str]:
    """N-gramas de caractere dao ao vetor denso a tolerancia morfologica que o BM25 nao tem."""
    padded = f"^{token}$"
    if len(padded) <= n:
        return [padded]
    return [padded[i : i + n] for i in range(len(padded) - n + 1)]


# ── Chunking ──────────────────────────────────────────────────────────────

class Chunk:
    """Text unit: a menor unidade citavel do pipeline."""

    __slots__ = ("chunk_id", "doc_id", "title", "text", "position", "tokens", "entities")

    def __init__(self, chunk_id: str, doc_id: str, title: str, text: str, position: int) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.position = position
        self.tokens: list[str] = tokenize(text)
        self.entities: list[str] = []

    def __repr__(self) -> str:  # pragma: no cover - debug
        return f"Chunk({self.chunk_id}, {len(self.tokens)} tokens)"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "entities": self.entities,
        }


def chunk_document(
    doc_id: str,
    title: str,
    text: str,
    *,
    target_tokens: int = 90,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Chunking por frase com alvo de tokens e sobreposicao.

    Cortar em fronteira de frase (e nao em janela fixa de tokens) preserva as
    relacoes sujeito-predicado que o extrator de entidades depende para inferir
    arestas do grafo — chunking ruim degrada o grafo antes de degradar o vetor.
    """
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0
    position = 0

    def flush() -> None:
        nonlocal buffer, buffer_len, position
        if not buffer:
            return
        body = " ".join(buffer).strip()
        chunks.append(Chunk(f"{doc_id}#c{position}", doc_id, title, body, position))
        position += 1
        buffer = buffer[-overlap_sentences:] if overlap_sentences else []
        buffer_len = sum(len(tokenize(s)) for s in buffer)

    for sentence in sentences(text):
        size = len(tokenize(sentence))
        if buffer and buffer_len + size > target_tokens:
            flush()
        buffer.append(sentence)
        buffer_len += size

    if buffer:
        body = " ".join(buffer).strip()
        # evita duplicar um chunk que seria apenas o overlap do anterior
        if not chunks or body != chunks[-1].text:
            chunks.append(Chunk(f"{doc_id}#c{position}", doc_id, title, body, position))
    return chunks
