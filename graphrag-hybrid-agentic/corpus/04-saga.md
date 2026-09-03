# Saga

Uma **Saga** coordena uma transacao de negocio que atravessa varios servicos, substituindo o commit distribuido por uma sequencia de transacoes locais com compensacoes. Se o passo tres falha, os passos dois e um sao desfeitos por acoes compensatorias explicitas.

Existem duas variantes. Na **Saga** coreografada, cada servico reage a eventos publicados no **Kafka** e nao existe coordenador; o acoplamento e' baixo, mas o fluxo fica implicito e dificil de depurar. Na **Saga** orquestrada, um coordenador central mantem a maquina de estados e chama cada participante; o fluxo fica legivel ao custo de um ponto de coordenacao.

A **Saga** nao oferece isolamento: estados intermediarios sao visiveis para outros leitores. Padroes como chave semantica de bloqueio e commutatividade de operacoes mitigam o problema.

Em sistemas agenticos, o orquestrador de agentes reproduz a variante orquestrada — o **LangGraph** e' essencialmente uma maquina de estados com compensacao manual.
