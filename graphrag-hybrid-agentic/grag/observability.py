"""
Observabilidade sem dependencias externas.

Modela o mesmo contrato do OpenTelemetry (spans aninhados, atributos, duracao)
usando apenas a stdlib, para que o exemplo rode em qualquer maquina. Quando o
pacote `opentelemetry-sdk` esta instalado, `bridge_to_otel()` reexporta os
spans coletados — ver ADR-006.

Alem dos traces, o Tracer contabiliza *custo*: chamadas de LLM em tempo de
indexacao versus em tempo de consulta. E' esse contador que sustenta a
comparacao LazyGraphRAG x GraphRAG classico na tabela de avaliacao.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class Span:
    name: str
    span_id: str
    parent_id: str | None
    start_ms: float
    end_ms: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    status: str = "ok"

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    @property
    def duration_ms(self) -> float:
        return round((self.end_ms or time.perf_counter() * 1000) - self.start_ms, 3)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """Coletor de spans com pilha explicita (thread-local nao e' necessario aqui)."""

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._stack: list[Span] = []
        self.counters: dict[str, float] = {}

    # ── spans ─────────────────────────────────────────────────────────────
    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        current = Span(
            name=name,
            span_id=uuid.uuid4().hex[:12],
            parent_id=self._stack[-1].span_id if self._stack else None,
            start_ms=time.perf_counter() * 1000,
            attributes=dict(attributes),
        )
        self.spans.append(current)
        self._stack.append(current)
        try:
            yield current
        except Exception as exc:  # pragma: no cover - caminho de erro
            current.status = f"error: {type(exc).__name__}"
            raise
        finally:
            current.end_ms = time.perf_counter() * 1000
            self._stack.pop()

    def event(self, message: str, **payload: Any) -> None:
        if self._stack:
            self._stack[-1].events.append({"message": message, **payload})

    # ── contadores de custo ───────────────────────────────────────────────
    def incr(self, counter: str, value: float = 1.0) -> None:
        self.counters[counter] = self.counters.get(counter, 0.0) + value

    def snapshot_counters(self) -> dict[str, float]:
        return dict(self.counters)

    def reset_counters(self, prefix: str | None = None) -> None:
        if prefix is None:
            self.counters.clear()
        else:
            for key in [k for k in self.counters if k.startswith(prefix)]:
                del self.counters[key]

    # ── exportacao ────────────────────────────────────────────────────────
    def tree(self) -> list[dict]:
        """Reconstroi a hierarquia de spans para leitura humana."""
        by_parent: dict[str | None, list[Span]] = {}
        for span in self.spans:
            by_parent.setdefault(span.parent_id, []).append(span)

        def build(parent_id: str | None) -> list[dict]:
            out = []
            for span in by_parent.get(parent_id, []):
                node = span.to_dict()
                node["children"] = build(span.span_id)
                out.append(node)
            return out

        return build(None)

    def export_json(self, path: str) -> None:
        payload = {"counters": self.counters, "traces": self.tree()}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def render(self, max_depth: int = 4) -> str:
        lines: list[str] = []

        def walk(nodes: list[dict], depth: int) -> None:
            if depth > max_depth:
                return
            for node in nodes:
                prefix = "  " * depth + ("└─ " if depth else "")
                detail = " ".join(f"{k}={v}" for k, v in node["attributes"].items())
                lines.append(f"{prefix}{node['name']} ({node['duration_ms']}ms) {detail}".rstrip())
                walk(node["children"], depth + 1)

        walk(self.tree(), 0)
        return "\n".join(lines)

    def bridge_to_otel(self, service_name: str = "graphrag-hybrid-agentic") -> bool:
        """Reexporta os spans coletados para o SDK do OpenTelemetry, se disponivel."""
        try:  # pragma: no cover - depende de pacote opcional
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        except ImportError:
            return False

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        otel = trace.get_tracer(service_name, tracer_provider=provider)
        for span in self.spans:
            with otel.start_as_current_span(span.name) as otel_span:
                for key, value in span.attributes.items():
                    otel_span.set_attribute(key, str(value))
        return True


_tracer = Tracer()


def get_tracer() -> Tracer:
    return _tracer


def reset_tracer() -> Tracer:
    global _tracer
    _tracer = Tracer()
    return _tracer
