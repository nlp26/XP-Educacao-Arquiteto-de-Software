"""
GraphRAG + Busca Hibrida Agentica — implementacao de referencia.

Ordem de leitura sugerida:
    config -> text -> embeddings -> retrievers   (recuperacao classica)
    extraction -> graph -> communities           (construcao do grafo)
    strategies                                   (local, global, DRIFT, lazy)
    agent                                        (roteamento adaptativo + critico)
"""
from .config import Settings, DEFAULT
from .pipeline import Index, build_index, load_corpus

__all__ = ["Settings", "DEFAULT", "Index", "build_index", "load_corpus"]
__version__ = "0.1.0"
