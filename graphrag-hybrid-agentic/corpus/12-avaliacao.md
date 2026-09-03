# Avaliacao de Recuperacao

Avaliar recuperacao exige separar duas perguntas. **Recall@k** mede se o documento relevante entrou na lista; **nDCG@k** mede se ele entrou em posicao alta. Otimizar so **nDCG** esconde falhas de cobertura, e otimizar so **Recall** esconde ruido no topo — as duas metricas precisam ser lidas juntas.

O **MRR** (Mean Reciprocal Rank) complementa quando existe uma unica resposta certa, medindo a posicao do primeiro acerto. Para respostas geradas, a metrica pratica e' a **Fundamentacao**: a fracao de afirmacoes da resposta que pode ser rastreada ate um trecho citado.

Toda comparacao de estrategia precisa incluir custo e **Latencia**, nao so qualidade. Uma estrategia que ganha dois pontos de **nDCG** custando cinco chamadas de LLM a mais por consulta pode ser a escolha errada. E' exatamente esse trade-off que separa **GraphRAG** classico de **LazyGraphRAG**.

O conjunto de avaliacao deve conter perguntas dos quatro tipos que o **Roteamento Adaptativo** distingue, senao o roteador e' medido apenas no caso facil.
