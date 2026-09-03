"""
Abstracao de LLM com tres backends e — crucialmente — um fallback deterministico.

O pipeline inteiro roda com `offline`, que substitui cada chamada de modelo por
uma heuristica extrativa. Isso nao e' um detalhe de conveniencia: e' a decisao
de arquitetura que torna o exemplo (a) reproduzivel bit a bit, (b) testavel em
CI sem chave nem GPU e (c) mensuravel — a tabela de avaliacao compara
estrategias de *recuperacao*, e um gerador estocastico no meio contaminaria a
medicao. Ver ADR-005.

Backends:
    offline    heuristica determinista (default)
    ollama     modelo local via HTTP (`ollama serve`)
    anthropic  API Claude (ANTHROPIC_API_KEY), default claude-sonnet-5
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from .observability import get_tracer
from .text import normalize, sentences, tokenize


class LLM:
    """Interface unica. `structured()` pede JSON; `complete()` pede texto livre."""

    name = "base"

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> str:
        raise NotImplementedError

    def structured(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> Any:
        raw = self.complete(prompt, system=system, max_tokens=max_tokens)
        return extract_json(raw)


def extract_json(raw: str) -> Any:
    """Tolerante a cercas de codigo e prosa em volta do JSON."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


class OfflineLLM(LLM):
    """
    Sumarizador extrativo. Nao "gera": seleciona e reordena frases do contexto.

    Efeito colateral desejavel — toda saida e' verificavel contra a fonte, entao
    o critico de groundedness mede a estrategia de recuperacao, nao a fluencia
    do modelo.
    """

    name = "offline"

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> str:
        get_tracer().incr("llm.calls")
        get_tracer().incr("llm.offline_calls")
        question, context = _split_prompt(prompt)
        # Os blocos de contexto vem rotulados ("[3] (Titulo) ..."); o rotulo
        # nao e' conteudo e atrapalharia tanto a selecao quanto a citacao.
        context = re.sub(r"\[\d+\]\s*\([^)]*\)\s*", "", context)
        query_terms = set(tokenize(question))
        ranked: list[tuple[float, str]] = []
        for sentence in sentences(context):
            terms = set(tokenize(sentence))
            if not terms:
                continue
            overlap = len(terms & query_terms) / (len(query_terms) or 1)
            density = len(terms & query_terms) / (len(terms) ** 0.5 or 1)
            ranked.append((overlap + 0.3 * density, sentence))
        ranked.sort(key=lambda item: -item[0])
        picked = [s for score, s in ranked[:4] if score > 0] or [s for _, s in ranked[:2]]
        return " ".join(picked).strip()

    def structured(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> Any:
        # Sem backend generativo nao ha JSON confiavel: quem chama deve ter
        # um caminho heuristico proprio. Retornar None e' o contrato.
        get_tracer().incr("llm.calls")
        get_tracer().incr("llm.offline_calls")
        return None


class OllamaLLM(LLM):
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434", temperature: float = 0.1) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> str:
        tracer = get_tracer()
        tracer.incr("llm.calls")
        tracer.incr("llm.remote_calls")
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": max_tokens},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data["message"]["content"].strip()


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")

    def available(self) -> bool:
        return bool(self.api_key)

    def complete(self, prompt: str, system: str | None = None, max_tokens: int = 800) -> str:
        tracer = get_tracer()
        tracer.incr("llm.calls")
        tracer.incr("llm.remote_calls")
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
        return "".join(block.get("text", "") for block in data.get("content", [])).strip()


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Separa pergunta e contexto nos prompts do pipeline (marcados por PERGUNTA:/CONTEXTO:)."""
    lowered = normalize(prompt)
    q_idx, c_idx = lowered.rfind("pergunta:"), lowered.find("contexto:")
    question = prompt[q_idx + len("pergunta:") :].strip() if q_idx != -1 else prompt
    if c_idx != -1:
        context_end = q_idx if q_idx > c_idx else len(prompt)
        context = prompt[c_idx + len("contexto:") : context_end].strip()
    else:
        context = prompt
    return question, context


def build_llm(backend: str, model: str) -> LLM:
    """Fabrica com degradacao graciosa: backend indisponivel cai para offline."""
    if backend == "ollama":
        llm = OllamaLLM(model=model)
        return llm if llm.available() else OfflineLLM()
    if backend == "anthropic":
        llm = AnthropicLLM(model=model if model.startswith("claude") else "claude-sonnet-5")
        return llm if llm.available() else OfflineLLM()
    return OfflineLLM()
