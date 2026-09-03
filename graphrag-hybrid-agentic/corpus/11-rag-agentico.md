# RAG Agentico e Roteamento Adaptativo

**RAG Agentico** substitui o pipeline de uma passada por um ciclo: o agente planeja, recupera, critica a propria saida e decide se recupera de novo. O ganho vem de perguntas multi-hop, em que a evidencia necessaria para o segundo salto so aparece depois do primeiro.

O componente central e' o **Roteamento Adaptativo**. Um classificador estima a complexidade da pergunta e escolhe o caminho: pergunta factual simples vai para **Busca Hibrida** direta, pergunta sobre relacoes entre entidades vai para **Local Search** no grafo, pergunta tematica vai para **Global Search**, e pergunta comparativa multi-hop vai para **DRIFT Search**. Rotear e' o que impede que o custo do modo mais caro seja pago em toda consulta.

A **Decomposicao de Consulta** quebra a pergunta em sub-perguntas independentes, recuperadas em paralelo e sintetizadas ao final. O **Critico** fecha o ciclo verificando fundamentacao e cobertura; quando reprova, o planejador amplia o top-k, troca de estrategia ou gera novas sub-perguntas.
