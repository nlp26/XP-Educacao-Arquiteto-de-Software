# CQRS

**CQRS** (Command Query Responsibility Segregation) separa o modelo de escrita do modelo de leitura. Comandos alteram estado e passam por invariantes de dominio; consultas leem projecoes desnormalizadas e nunca modificam nada. A separacao permite escalar leitura e escrita de forma independente, com bancos diferentes para cada lado.

O preco e' a **consistencia eventual**: a projecao de leitura fica atras do modelo de escrita por alguns milissegundos ou segundos. Interfaces que exibem o resultado imediato de um comando precisam de leitura pos-escrita explicita, senao o usuario ve dado velho.

**CQRS** combina naturalmente com **Event Sourcing**, porque os eventos ja sao o mecanismo de propagacao para as projecoes. Mas os dois padroes sao independentes: e' possivel aplicar **CQRS** sobre um banco relacional classico usando views materializadas e o padrao **Outbox** para publicar as mudancas.

O antipadrao mais comum e' aplicar **CQRS** ao sistema inteiro. O custo operacional so se paga em contextos com assimetria real entre leitura e escrita.
