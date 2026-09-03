# ADR-006 — Observabilidade propria compativel com OpenTelemetry

**Status**: aceito

## Contexto

O ciclo agentico precisa ser inspecionavel: qual estrategia rodou, quantas iteracoes,
o que o critico reprovou, quantas chamadas de LLM cada consulta custou. O padrao da
industria e' OpenTelemetry, ja adotado no prototipo multiagente deste repositorio
(ADR-003). Mas o SDK do OTel traz varias dependencias e um coletor para ser util, o que
contraria o objetivo de execucao imediata.

## Decisao

Implementar `grag/observability.py` com o **mesmo contrato conceitual** do OTel — spans
aninhados com atributos, eventos e duracao — usando apenas a stdlib, mais um metodo
`bridge_to_otel()` que reexporta os spans coletados quando `opentelemetry-sdk` esta
instalado.

Alem dos spans, o tracer mantem **contadores de custo** (`llm.calls`, `index.reports`),
separando custo de indexacao de custo de consulta.

## Consequencias

**Positivas**
- `--trace` imprime a arvore de execucao sem instalar nada.
- Os contadores tornam mensuravel o trade-off central do GraphRAG: LazyGraphRAG nao gera
  Community Reports, e isso aparece como numero, nao como afirmacao.
- A migracao para OTel real e' aditiva, nao uma reescrita.

**Negativas**
- Sem propagacao de contexto entre processos nem exporters (OTLP, Jaeger).
- O tracer e' um singleton de modulo: em execucao concorrente seria necessario torna-lo
  thread-local.

## Alternativas consideradas

| Alternativa | Por que nao |
|---|---|
| OTel SDK obrigatorio | contraria o requisito de zero dependencias |
| `logging` puro | perde a estrutura de arvore, que e' o que torna o ciclo agentico legivel |
| Sem instrumentacao | impossivel justificar escolhas de estrategia com evidencia |
