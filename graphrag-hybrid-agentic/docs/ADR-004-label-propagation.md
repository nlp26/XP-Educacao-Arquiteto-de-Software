# ADR-004 — Label Propagation no lugar do Leiden para deteccao de comunidades

**Status**: aceito
**Contexto**: implementacao de referencia GraphRAG + busca hibrida agentica

## Contexto

O GraphRAG de referencia (Microsoft Research) detecta comunidades com o algoritmo
**Leiden**, que otimiza modularidade e produz hierarquia nativa. Leiden depende de
`igraph`/`leidenalg` — extensoes C com build pesado — e e' estocastico: duas execucoes
com sementes diferentes produzem particoes diferentes.

Este projeto tem dois requisitos que colidem com isso: rodar sem nenhuma dependencia
externa e produzir numeros identicos em qualquer maquina, porque a tabela de avaliacao
compara estrategias de recuperacao e nao pode ter ruido de particionamento no meio.

## Decisao

Usar **Label Propagation** com desempate lexical deterministico, aplicado em cascata
sobre o grafo contraido para obter os niveis hierarquicos (`grag/graph.py`).

## Consequencias

**Positivas**
- Zero dependencias; O(E) por iteracao; particao identica a cada execucao.
- A hierarquia por contracao reproduz a interface que o Global Search espera
  (comunidades por nivel, com `parent`).

**Negativas**
- Label Propagation nao otimiza modularidade explicitamente: em grafos com hubs muito
  conectados ele tende a produzir uma comunidade gigante. No corpus de referencia isso
  nao ocorreu (17 comunidades no nivel 0 para 45 entidades), mas ocorreria em escala.
- Nao ha controle de resolucao (o `resolution` do Leiden), entao o numero de comunidades
  nao e' ajustavel.

**Mitigacao**: a interface `detect_communities()` retorna `dict[int, list[Community]]` e
nada mais no pipeline depende do algoritmo. Trocar por Leiden e' substituir um metodo.

## Alternativas consideradas

| Alternativa | Por que nao |
|---|---|
| Leiden (`leidenalg`) | dependencia C + estocasticidade; e' a escolha certa em producao |
| Girvan-Newman | O(V·E²), inviavel acima de alguns milhares de nos |
| Louvain puro | mesma familia do Leiden, sem a garantia de comunidades bem conectadas |
