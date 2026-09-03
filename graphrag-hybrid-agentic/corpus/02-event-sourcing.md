# Event Sourcing

**Event Sourcing** armazena a sequencia de eventos imutaveis como fonte da verdade, em vez do estado atual. O estado e' derivado pela reproducao dos eventos, e snapshots periodicos evitam replays longos.

A vantagem decisiva e' a auditoria: todo estado tem historia explicita, e o sistema pode responder "como chegamos aqui" sem tabelas de log paralelas. **Event Sourcing** tambem habilita projecoes retroativas — uma nova visao de leitura e' construida reprocessando o log desde o inicio.

O custo esta na evolucao de esquema. Eventos antigos permanecem para sempre, entao versionamento de evento e upcasting deixam de ser opcionais. Deletar dado pessoal exige criptografia por chave descartavel, ja que o log e' imutavel.

O **Event Store** normalmente e' um append-only sobre **PostgreSQL** ou um log distribuido como **Kafka**. Em ambos os casos, o padrao **Outbox** garante que a escrita transacional e a publicacao do evento nao divirjam.
