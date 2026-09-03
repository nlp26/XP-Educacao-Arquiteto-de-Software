# Padrao Outbox

O padrao **Outbox** resolve a escrita dual: gravar no banco e publicar no broker sao duas operacoes que nao compartilham transacao. Sem o padrao, uma falha entre elas deixa o sistema inconsistente — o pedido existe mas o evento nunca chegou ao **Kafka**.

A solucao grava o evento numa tabela `outbox` dentro da mesma transacao do dado de negocio. Um processo separado le a tabela e publica no broker, marcando a linha como enviada. A leitura pode usar polling ou **Change Data Capture** sobre o log de replicacao do **PostgreSQL**.

A entrega resultante e' *at-least-once*: o mesmo evento pode ser publicado duas vezes se o processo cair depois de publicar e antes de marcar. Consumidores precisam ser idempotentes, tipicamente por chave de deduplicacao.

**Outbox** e' pre-requisito pratico para **Event Sourcing** e para projecoes de **CQRS** confiaveis em qualquer arquitetura de **Microservicos**.
