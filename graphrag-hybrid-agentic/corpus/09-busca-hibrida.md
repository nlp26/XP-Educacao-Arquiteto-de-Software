# Busca Hibrida

A **Busca Hibrida** executa em paralelo uma perna lexical e uma perna densa, e funde os resultados. A perna lexical usa **BM25**, que acerta termos raros, codigos e nomes proprios exatos. A perna densa usa embeddings sobre um indice **HNSW**, que acerta parafrase e sinonimo. As duas erram em situacoes complementares, e e' por isso que a fusao ganha das duas isoladas.

A fusao padrao e' a **RRF** (Reciprocal Rank Fusion), que soma o inverso da posicao de cada documento em cada lista com uma constante de amortecimento, tipicamente 60. Por operar sobre posicao e nao sobre score, a **RRF** e' imune a escalas incomparaveis entre **BM25** e similaridade de cosseno.

A alternativa e' a **Combinacao Convexa**, que normaliza os scores de cada perna e aplica um peso alpha. Preserva a margem entre o primeiro e o segundo colocado, o que a **RRF** descarta, mas exige recalibrar alpha quando o corpus muda.
