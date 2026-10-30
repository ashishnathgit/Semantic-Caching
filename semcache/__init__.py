"""Semantic cache for natural-language queries, built on linear algebra.

Package layout
--------------
linalg     every matrix operation (TF-IDF, QR, Jacobi, SVD, cosine, LSH), from scratch
text       tokenizer and vocabulary
embedder   LSA embedder: text -> TF-IDF row -> latent vector (calls linalg)
cache      the semantic cache: threshold lookup, LRU eviction, persistence (calls linalg)
backends   who answers on a miss: Ollama, or a stub fallback
pipeline   SemanticCacheService.ask() ties the pieces together
data       FAQ and Quora Question Pairs loaders
evaluate   threshold / rank sweeps, latency benchmark, plots
cli        command-line entry point (python -m semcache ...)
"""

__version__ = "1.0.0"
