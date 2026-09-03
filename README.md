# XP-Educação-Arquiteto-de-Software
Fundamentos de Arquitetura de Software. Requisitos Arquiteturais e Modelagem Arquitetural. Design Patterns, Estilos e Padrões Arquiteturais. Principais Arquiteturas de Software da Atualidade.

## Projetos deste repositório

| Projeto | Descrição |
|---|---|
| [`aruitetura-api-final/`](aruitetura-api-final) | API REST de clientes em FastAPI, padrão MVC com PostgreSQL — desafio final do bootcamp |
| [`multiagent_prototype/`](multiagent_prototype) | Protótipo de arquitetura de referência multiagente com LangGraph, memória híbrida e OpenTelemetry (ADRs #001–#003) |
| [`graphrag-hybrid-agentic/`](graphrag-hybrid-agentic) | GraphRAG (Local, Global, DRIFT, LazyGraphRAG) + busca híbrida (BM25 + denso, RRF, reranking, MMR) com roteamento adaptativo e avaliação medida — roda sem dependências (ADRs #004–#008) |

---

# Clientes API

API REST para gestao de clientes, construida com **Python/FastAPI** no padrao arquitetural **MVC**, com persistencia em **PostgreSQL**.

> Desafio Final — Bootcamp Arquiteto(a) de Software

---

## Stack

| Componente      | Tecnologia           |
|-----------------|----------------------|
| Linguagem       | Python 3.12          |
| Framework       | FastAPI 0.115        |
| ORM             | SQLAlchemy 2.0       |
| Banco de dados  | PostgreSQL 16        |
| Validacao       | Pydantic v2          |
| Infra           | Docker + Compose     |

---

## Arquitetura

O projeto segue o padrao **MVC** adaptado para API REST (sem camada de View com templates — a "View" eh o JSON serializado pelo Pydantic).

### Diagrama C4

#### Level 1 — Contexto do sistema

#### Level 3 — Componentes (MVC)

> O arquivo editavel `docs/arquitetura-c4.drawio` pode ser aberto no [draw.io](https://app.diagrams.net/) ou no VS Code com a extensao Draw.io Integration.

### Fluxo de um request

```
HTTP Request (JSON)
       │
       ▼
 Controller ──→ Schema (valida entrada)
       │
       ▼
   Service (regras de negocio)
       │
       ▼
  Repository (queries)
       │
       ▼
 Model / ORM ──→ PostgreSQL
```

---

## Estrutura de Pastas

```
clientes-api/
├── app/
│   ├── __init__.py
│   ├── main.py                          # Entry point — cria o FastAPI e registra routers
│   ├── config.py                        # Carrega DATABASE_URL do ambiente
│   ├── database.py                      # Engine, Session factory, dependency get_db()
│   │
│   ├── controller/
│   │   └── cliente_controller.py        # Rotas REST (GET, POST, PUT, DELETE)
│   │
│   ├── service/
│   │   └── cliente_service.py           # Logica de negocio e validacoes
│   │
│   ├── repository/
│   │   └── cliente_repository.py        # Acesso a dados, encapsula queries
│   │
│   ├── model/
│   │   └── cliente.py                   # Entidade ORM → tabela 'clientes'
│   │
│   └── schema/
│       └── cliente_schema.py            # DTOs Pydantic (entrada e saida)
│
├── docs/
│   └── arquitetura-c4.drawio            # Diagramas C4 (contexto + componentes)
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

### Papel de cada componente

**Controller** (`controller/cliente_controller.py`)
Camada de apresentacao. Define os endpoints REST usando `APIRouter`. Recebe o request HTTP, delega pro Service e retorna a resposta. Nao tem logica de negocio.

**Service** (`service/cliente_service.py`)
Regras de negocio. Valida se o recurso existe antes de atualizar/deletar, lanca HTTP 404 quando necessario, converte DTOs em entidades. Orquestra chamadas ao Repository.

**Repository** (`repository/cliente_repository.py`)
Acesso a dados. Encapsula todas as queries ao banco (find_all, find_by_id, find_by_nome, count, save, update, delete). Se precisar trocar o banco, so essa camada muda.

**Model** (`model/cliente.py`)
Entidade de dominio. Classe `Cliente` mapeada via SQLAlchemy para a tabela `clientes`. Define campos `id`, `nome`, `email`, `telefone`.

**Schema** (`schema/cliente_schema.py`)
DTOs Pydantic. `ClienteCreate` valida o payload de entrada, `ClienteResponse` formata a saida. Separa a representacao HTTP da representacao interna do ORM.

---

## Endpoints

| Metodo   | Rota                     | Descricao                              |
|----------|--------------------------|----------------------------------------|
| GET      | `/clientes/`             | Listar todos os clientes               |
| GET      | `/clientes/contar`       | Total de registros                     |
| GET      | `/clientes/buscar?nome=` | Buscar por nome (parcial, ilike)       |
| GET      | `/clientes/{id}`         | Buscar por ID                          |
| POST     | `/clientes/`             | Criar novo cliente                     |
| PUT      | `/clientes/{id}`         | Atualizar cliente                      |
| DELETE   | `/clientes/{id}`         | Remover cliente                        |
| GET      | `/health`                | Health check                           |

---

## Como rodar

### Docker (recomendado)

```bash
git clone https://github.com/nlp2026/clientes-api.git
cd clientes-api
docker compose up --build
```

API disponivel em `http://localhost:8000`
Swagger em `http://localhost:8000/docs`

### Local (sem Docker)

Prerequisito: PostgreSQL rodando na porta 5432.

```bash
createdb -U postgres clientes_db
cp .env.example .env
# ajustar DATABASE_URL se necessario

pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Exemplos (cURL)

```bash
# Criar
curl -X POST http://localhost:8000/clientes/ \
  -H "Content-Type: application/json" \
  -d '{"nome": "Maria Silva", "email": "maria@email.com", "telefone": "11999990000"}'

# Listar todos
curl http://localhost:8000/clientes/

# Buscar por nome
curl "http://localhost:8000/clientes/buscar?nome=maria"

# Contar
curl http://localhost:8000/clientes/contar

# Buscar por ID
curl http://localhost:8000/clientes/1

# Atualizar
curl -X PUT http://localhost:8000/clientes/1 \
  -H "Content-Type: application/json" \
  -d '{"nome": "Maria Santos", "email": "maria@email.com", "telefone": "11988880000"}'

# Deletar
curl -X DELETE http://localhost:8000/clientes/1
```

---

## Persistencia

- **Banco**: PostgreSQL 16 via Docker
- **ORM**: SQLAlchemy 2.0 com `declarative_base`
- **Session**: uma por request via `get_db()` (dependency injection do FastAPI)
- **Criacao de tabelas**: `Base.metadata.create_all()` no startup (em producao, usar Alembic)
- **Volume**: `pg_data` persiste dados entre restarts do container
