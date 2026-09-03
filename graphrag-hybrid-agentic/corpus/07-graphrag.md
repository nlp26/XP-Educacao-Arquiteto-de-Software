# GraphRAG

**GraphRAG** indexa um corpus construindo um grafo de conhecimento antes de responder. A ingestao extrai entidades e relacoes de cada text unit com um LLM, consolida entidades equivalentes e detecta comunidades com o algoritmo **Leiden**. Cada comunidade recebe um **Community Report** — um resumo gerado que descreve o tema daquele subgrafo.

Existem dois modos classicos de consulta. O **Local Search** ancora a pergunta em entidades especificas, expande pela vizinhanca do grafo e recupera os text units associados; e' o modo certo para perguntas sobre entidades nomeadas. O **Global Search** ignora entidades individuais e faz map-reduce sobre os **Community Reports**, respondendo perguntas tematicas do tipo "quais sao os temas centrais do corpus".

O gargalo do **GraphRAG** e' o custo de indexacao: uma chamada de LLM por chunk para extracao, mais uma por comunidade para o relatorio. Em corpora grandes esse custo domina o orcamento inteiro do sistema.
