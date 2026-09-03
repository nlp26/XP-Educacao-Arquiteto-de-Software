"""
Testes do pipeline. Rodam com a stdlib:

    python -m unittest discover -s tests -v

O foco esta nas propriedades que, se quebrarem, invalidam a avaliacao:
determinismo, correcao das metricas e degradacao graciosa das estrategias de
grafo quando o entity linking falha.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import load_goldset, ndcg_at_k, recall_at_k, reciprocal_rank
from grag.agent import GraphRAGAgent, critique
from grag.config import Settings
from grag.embeddings import HashingEmbedder, cosine
from grag.pipeline import build_index
from grag.retrievers import Scored, mmr, reciprocal_rank_fusion
from grag.strategies import STRATEGIES, run_strategy
from grag.text import chunk_document, tokenize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "corpus")


def build() -> tuple:
    settings = Settings()
    settings.retrieval.top_k = 4
    return build_index(settings, corpus_dir=CORPUS), settings


class TestText(unittest.TestCase):
    def test_tokenize_remove_acentos_e_stopwords(self):
        tokens = tokenize("A observabilidade distribuída e o tracing")
        self.assertNotIn("a", tokens)
        self.assertTrue(any(t.startswith("observabil") for t in tokens))

    def test_chunking_respeita_sobreposicao(self):
        text = "Primeira frase. Segunda frase. Terceira frase. Quarta frase."
        chunks = chunk_document("d", "T", text, target_tokens=4, overlap_sentences=1)
        self.assertGreater(len(chunks), 1)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertTrue(
                set(tokenize(previous.text)) & set(tokenize(current.text)),
                "chunks consecutivos devem compartilhar a frase de sobreposicao",
            )

    def test_chunk_ids_sao_unicos(self):
        chunks = chunk_document("d", "T", "Uma. Duas. Tres. Quatro. Cinco.", target_tokens=2)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))


class TestEmbeddings(unittest.TestCase):
    def test_vetores_normalizados(self):
        embedder = HashingEmbedder(dim=128)
        embedder.fit([tokenize("evento broker kafka"), tokenize("consulta leitura escrita")])
        vector = embedder.encode("evento no broker")
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in vector)), 1.0, places=6)

    def test_similaridade_maior_para_texto_relacionado(self):
        embedder = HashingEmbedder(dim=256)
        docs = ["broker de eventos e mensageria", "modelo de leitura e escrita separados"]
        embedder.fit([tokenize(d) for d in docs])
        query = embedder.encode("mensageria com eventos")
        self.assertGreater(
            cosine(query, embedder.encode(docs[0])), cosine(query, embedder.encode(docs[1]))
        )


class TestFusion(unittest.TestCase):
    def test_rrf_premia_consenso_entre_pernas(self):
        legs = {
            "bm25": [Scored("a", 9.0), Scored("b", 8.0)],
            "dense": [Scored("b", 0.9), Scored("c", 0.8)],
        }
        fused = reciprocal_rank_fusion(legs, k=60)
        self.assertEqual(fused[0].chunk_id, "b", "documento presente nas duas listas deve liderar")

    def test_rrf_ponderada_respeita_o_peso(self):
        legs = {"main": [Scored("a", 1.0)], "sub": [Scored("b", 1.0)]}
        fused = reciprocal_rank_fusion(legs, k=60, weights={"main": 3.0, "sub": 1.0})
        self.assertEqual(fused[0].chunk_id, "a")

    def test_mmr_reduz_redundancia(self):
        index, _ = build()
        candidates = [Scored(cid, score) for score, cid in zip(range(9, 0, -1), list(index.chunks)[:9])]
        selected = mmr(candidates, index.dense, top_k=3, lambda_=0.2)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({s.chunk_id for s in selected}), 3)


class TestGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index, cls.settings = build()

    def test_grafo_tem_entidades_e_arestas(self):
        self.assertGreater(len(self.index.graph.entities), 20)
        self.assertGreater(len(self.index.graph.relations), 20)

    def test_resolucao_de_entidades_funde_sigla_e_forma_extensa(self):
        entities = self.index.graph.entities
        self.assertIn("CQRS", entities)
        self.assertNotIn("Command Query Responsibility Segregation", entities)

    def test_entity_linking_ignora_pontuacao(self):
        linked = self.index.graph.link_entities("Qual a relacao entre CQRS e Event Sourcing?")
        self.assertIn("CQRS", linked)
        self.assertIn("Event Sourcing", linked)
        self.assertEqual(len(linked), len(set(linked)), "sem duplicatas de alias")

    def test_ppr_concentra_massa_perto_da_semente(self):
        ranks = self.index.graph.personalized_pagerank(["Outbox"])
        self.assertEqual(max(ranks, key=ranks.get), "Outbox")
        vizinhos = set(self.index.graph.neighbors("Outbox"))
        distantes = [n for n in ranks if n not in vizinhos and n != "Outbox"]
        if vizinhos and distantes:
            self.assertGreater(
                max(ranks[n] for n in vizinhos), min(ranks[n] for n in distantes)
            )

    def test_comunidades_particionam_todas_as_entidades(self):
        for level, communities in self.index.graph.communities.items():
            membros = [m for c in communities for m in c.members]
            self.assertEqual(
                sorted(membros), sorted(self.index.graph.entities),
                f"nivel {level} deve cobrir cada entidade exatamente uma vez",
            )


class TestStrategies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index, cls.settings = build()

    def test_todas_as_estrategias_retornam_contexto(self):
        query = "Como o padrao Outbox garante a entrega no Kafka?"
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                retrieval = run_strategy(name, self.index, query, self.settings)
                self.assertTrue(retrieval.context, f"{name} retornou contexto vazio")
                self.assertTrue(all(c.chunk_id in self.index.chunks for c in retrieval.context))

    def test_local_search_degrada_para_hibrida_sem_ancora(self):
        retrieval = run_strategy("local_search", self.index, "xyzzy plugh foobar", self.settings)
        self.assertEqual(retrieval.seed_entities, [])
        self.assertIsInstance(retrieval.context, list)

    def test_drift_gera_perguntas_de_acompanhamento(self):
        retrieval = run_strategy(
            "drift_search", self.index, "Compare DRIFT Search e LazyGraphRAG", self.settings
        )
        self.assertTrue(retrieval.subqueries)

    def test_lazy_nao_usa_community_reports(self):
        retrieval = run_strategy("lazy_graphrag", self.index, "O que e Event Sourcing?", self.settings)
        self.assertEqual(retrieval.communities, [])


class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index, cls.settings = build()
        cls.agent = GraphRAGAgent(cls.index, cls.settings)

    def test_roteamento_por_tipo_de_pergunta(self):
        casos = [
            ("O que faz o Circuit Breaker?", "factual"),
            ("Qual a relacao entre CQRS e Event Sourcing?", "relacional"),
            ("Quais sao os temas centrais do corpus?", "tematica"),
            ("Compare DRIFT Search e LazyGraphRAG", "comparativa"),
        ]
        for query, esperado in casos:
            with self.subTest(query=query):
                self.assertEqual(self.agent.router.plan(query).kind, esperado)

    def test_resposta_cita_apenas_chunks_recuperados(self):
        result = self.agent.answer("Como o padrao Outbox garante a entrega no Kafka?")
        self.assertTrue(result.citations)
        for citation in result.citations:
            self.assertIn(citation["chunk_id"], self.index.chunks)

    def test_critico_reprova_resposta_sem_fundamentacao(self):
        retrieval = run_strategy("hybrid", self.index, "O que e CQRS?", self.settings)
        verdict = critique(
            self.index,
            "O que e CQRS?",
            "Zebras coloridas atravessam o deserto em bicicletas amarelas.",
            [{"chunk_id": retrieval.context[0].chunk_id}],
            retrieval,
            self.settings,
        )
        self.assertFalse(verdict.passed)
        self.assertLess(verdict.groundedness, self.settings.agent.groundedness_threshold)

    def test_ciclo_respeita_o_limite_de_iteracoes(self):
        result = self.agent.answer("qwertz plugh xyzzy")
        self.assertLessEqual(result.iterations, self.settings.agent.max_iterations)


class TestMetrics(unittest.TestCase):
    def test_recall_e_ndcg_com_valores_conhecidos(self):
        relevance = {"a": 2, "b": 1}
        self.assertAlmostEqual(recall_at_k(["a", "x", "b"], relevance, 3), 1.0)
        self.assertAlmostEqual(recall_at_k(["a", "x"], relevance, 2), 0.5)
        # ideal: 2/log2(2) + 1/log2(3); obtido: 1/log2(2) + 2/log2(3)
        obtido = 1 / math.log2(2) + 2 / math.log2(3)
        ideal = 2 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(ndcg_at_k(["b", "a"], relevance, 2), obtido / ideal)
        self.assertAlmostEqual(reciprocal_rank(["x", "b"], relevance), 0.5)

    def test_goldset_referencia_apenas_chunks_existentes(self):
        index, _ = build()
        for item in load_goldset():
            for chunk_id in item["relevance"]:
                self.assertIn(chunk_id, index.chunks, f"{item['id']} cita chunk inexistente")


class TestDeterminism(unittest.TestCase):
    def test_duas_indexacoes_produzem_o_mesmo_grafo(self):
        primeira, settings = build()
        segunda, _ = build()
        self.assertEqual(sorted(primeira.graph.entities), sorted(segunda.graph.entities))
        self.assertEqual(sorted(primeira.graph.relations), sorted(segunda.graph.relations))
        query = "Qual a relacao entre CQRS e Event Sourcing?"
        self.assertEqual(
            run_strategy("local_search", primeira, query, settings).chunk_ids,
            run_strategy("local_search", segunda, query, settings).chunk_ids,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
