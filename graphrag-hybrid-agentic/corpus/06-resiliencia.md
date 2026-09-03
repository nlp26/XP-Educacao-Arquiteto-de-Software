# Resiliencia

O **Circuit Breaker** interrompe chamadas a uma dependencia que esta falhando, evitando que timeouts acumulados esgotem o pool de conexoes do chamador. Tem tres estados: fechado, aberto e meio-aberto, com uma janela de teste antes de restabelecer o trafego.

O **Bulkhead** isola recursos por dependencia — pools separados garantem que a lentidao de um servico nao consuma toda a capacidade do processo. Combinado com **timeout** agressivo e **retry** com backoff exponencial e jitter, forma o nucleo minimo de resiliencia em **Microservicos**.

Retry sem jitter cria tempestade sincronizada: todos os clientes voltam ao mesmo tempo e derrubam o servico recem-recuperado. Retry sem idempotencia duplica efeitos — motivo pelo qual consumidores de **Kafka** precisam de chave de deduplicacao.

Em pipelines de **RAG** agentico, os mesmos padroes se aplicam a chamadas de LLM: **timeout**, **Circuit Breaker** por provedor e degradacao graciosa para um modelo local via **Ollama**.
