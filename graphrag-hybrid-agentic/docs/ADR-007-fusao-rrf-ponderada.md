# ADR-007 — RRF ponderada como fusao padrao da busca hibrida

**Status**: aceito

## Contexto

A busca hibrida executa pernas com escalas incomparaveis: BM25 produz scores nao
limitados (0 a ~20 neste corpus), similaridade de cosseno vive em [-1, 1], e a perna de
grafo produz massa de PageRank normalizada. Somar isso diretamente e' indefensavel.

Duas fusoes sao praticadas em 2026: **RRF**, que opera sobre posicao, e **combinacao
convexa**, que normaliza scores por perna e aplica um peso `alpha`. Publicacoes recentes
mostram vantagem da combinacao convexa em alguns benchmarks, e da RRF em outros —
o resultado depende de quao calibradas as pernas estao.

## Decisao

Adotar **RRF ponderada** (`score = Σ w_l / (k + rank_l)`, com `k = 60`) como padrao,
mantendo a combinacao convexa selecionavel por configuracao (`retrieval.fusion`).

O peso `w_l` existe porque nem toda perna e' um par: no DRIFT Search, a consulta original
vale o dobro de cada pergunta de acompanhamento — sem peso, tres follow-ups genericos
vencem por maioria a evidencia que responde a pergunta.

## Consequencias

**Positivas**
- Imune a recalibracao: trocar o encoder denso nao exige re-tunar `alpha`.
- O peso resolve o modo de falha real do DRIFT e da decomposicao de consulta.
- Varredura no corpus de referencia (`evidence/sweep.md`): RRF e combinacao convexa
  ficam a menos de 0,01 de nDCG@3 uma da outra — nao ha vantagem que justifique o custo
  de calibracao da convexa neste corpus.

**Negativas**
- A RRF descarta a magnitude: quando uma perna esta muito mais certa que a outra, essa
  confianca se perde. Em corpora onde o denso e' claramente superior, a convexa ganha.
- `k = 60` e' herdado da literatura, nao ajustado para este corpus.

## Alternativas consideradas

| Alternativa | Por que nao |
|---|---|
| Combinacao convexa como padrao | exige recalibrar `alpha` a cada troca de encoder |
| Soma de scores brutos | matematicamente indefensavel entre escalas diferentes |
| Cascata (denso filtra, BM25 reordena) | perde recall do que o denso nao trouxe |
