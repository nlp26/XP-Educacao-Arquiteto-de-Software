# ADR-005 — Backend de LLM offline e deterministico como padrao

**Status**: aceito

## Contexto

Um exemplo de GraphRAG precisa de LLM em quatro pontos: extracao de entidades, geracao
de Community Reports, classificacao da consulta e sintese da resposta. Exigir um modelo
para rodar o exemplo cria tres problemas: nao roda em CI, nao roda sem GPU ou sem chave,
e — o mais grave — torna a avaliacao nao reproduzivel, porque a variancia do gerador se
mistura a diferenca entre estrategias de recuperacao.

## Decisao

Implementar `OfflineLLM` como **padrao**: um sumarizador extrativo deterministico que
seleciona e reordena frases do proprio contexto. Cada ponto do pipeline que chamaria um
LLM tem um caminho heuristico equivalente (gazetteer para extracao, resumo extrativo
para relatorios, marcadores lexicais para classificacao).

Os backends `ollama` e `anthropic` continuam disponiveis por variavel de ambiente
(`GRAG_LLM`), e a fabrica degrada para `offline` quando o backend escolhido nao responde.

## Consequencias

**Positivas**
- `python run_evidence.py` roda em ~3 s, sem rede, com numeros identicos em qualquer maquina.
- A metrica de fundamentacao mede a **recuperacao**, nao a fluencia do gerador: como a
  resposta e' extrativa, toda afirmacao e' rastreavel ate um trecho citado por construcao.
- A comparacao entre estrategias fica limpa — a unica variavel e' a recuperacao.

**Negativas**
- As respostas nao sao fluentes: sao frases coladas. O exemplo demonstra recuperacao e
  orquestracao, nao qualidade de redacao.
- Perguntas que exigem sintese real entre trechos (nao selecao) ficam sub-atendidas no
  modo offline; e' preciso ligar um backend para avaliar isso.

## Alternativas consideradas

| Alternativa | Por que nao |
|---|---|
| Exigir Ollama | inviabiliza CI e a reprodutibilidade dos numeros publicados |
| Modelo pequeno embarcado | dependencia de centenas de MB e ainda estocastico |
| Mock que retorna texto fixo | nao exercita nada; o extrativo ao menos depende do contexto real |
