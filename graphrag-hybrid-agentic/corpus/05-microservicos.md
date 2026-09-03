# Microservicos e Monolito Modular

**Microservicos** decompoem o sistema em servicos independentes com banco proprio e ciclo de deploy autonomo. O ganho e' organizacional antes de ser tecnico: equipes liberam sem coordenacao. O custo e' operacional — rede, versionamento de contrato, observabilidade distribuida e consistencia eventual entram todos de uma vez.

O **Monolito Modular** entrega grande parte do beneficio de fronteira sem o custo de rede. Modulos comunicam-se por interfaces internas e o banco e' compartilhado, mas particionado por schema. E' o ponto de partida recomendado quando as fronteiras de dominio ainda nao estao estaveis.

A migracao de monolito para **Microservicos** costuma usar o padrao **Strangler Fig**: um proxy roteia gradualmente rotas do sistema antigo para o novo. Cada rota migrada leva consigo seus dados, o que exige **Outbox** ou **Change Data Capture** durante a coexistencia.

O **Circuit Breaker** e o **Bulkhead** tornam-se obrigatorios: em rede, a falha parcial e' o estado normal.
