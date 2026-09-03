# Reranking e Diversidade

O **Reranker** e' um cross-encoder que le a consulta e o documento juntos, produzindo um score de relevancia muito mais preciso que o bi-encoder da busca vetorial. O custo e' quadratico em atencao, entao ele so roda sobre o topo da lista fundida — tipicamente entre 20 e 200 candidatos.

O ganho tipico do **Reranker** aparece em **nDCG**, nao em **Recall**: ele reordena o que a recuperacao ja trouxe e nao consegue resgatar o documento que ficou fora dos candidatos. Recuperacao ruim nao e' consertada por reranqueamento.

O **MMR** (Maximal Marginal Relevance) resolve um problema diferente: redundancia. Ele penaliza o candidato que e' muito parecido com o que ja foi selecionado, trocando um pouco de relevancia por cobertura. Em **GraphRAG** o **MMR** importa mais que em **RAG** plano, porque a expansao pelo grafo traz varios trechos quase identicos sobre a mesma entidade e o orcamento de contexto se esgota antes de cobrir a pergunta inteira.
