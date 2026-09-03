# DRIFT Search e LazyGraphRAG

O **DRIFT Search** combina os dois modos do **GraphRAG** numa unica consulta. Ele comeca por um primer global sobre os **Community Reports**, gera uma resposta hipotetica e, a partir dela, deriva perguntas de acompanhamento. Cada pergunta vira uma **Local Search**, cujo resultado alimenta a proxima rodada. O refinamento iterativo cobre perguntas que nao sao nem puramente locais nem puramente tematicas.

O **LazyGraphRAG** ataca o custo pelo outro lado: nao gera **Community Reports** na ingestao. O grafo e' construido com extracao barata de conceitos, e todo o trabalho caro e' adiado para o momento da consulta, quando apenas o subgrafo relevante e' explorado e resumido. O resultado publicado pela Microsoft Research reduz o custo de indexacao em cerca de tres ordens de grandeza mantendo a qualidade de resposta.

A escolha entre **DRIFT Search** e **LazyGraphRAG** e' uma escolha de onde pagar: indexacao cara e consulta barata, ou indexacao barata e consulta cara.
