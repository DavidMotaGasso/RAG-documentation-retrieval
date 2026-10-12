"""Local embedding model wrapper (BAAI/bge-small-en-v1.5 via sentence-transformers).

Documents (chunks) and queries are embedded differently: bge was trained to
match a short query prefixed with a search instruction against plain passages.
Vectors are L2-normalized, so cosine similarity equals the dot product.

Usage:
    embedder = Embedder()
    vectors = embedder.embed_documents(["chunk text", ...])
    query_vector = embedder.embed_query("Which dataset does the YOLOv11 paper use?")
"""

from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Recommended by the bge authors for short-query -> passage retrieval.
# Its effect on this corpus is checked in Phase 1.5.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME, use_query_instruction: bool = True):
        self.model_name = model_name
        self.use_query_instruction = use_query_instruction

    @cached_property
    def model(self) -> SentenceTransformer:
        # Loaded lazily: importing this module should not cost ~10 s.
        return SentenceTransformer(self.model_name, device="cpu")

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    @property
    def max_tokens(self) -> int:
        return self.model.max_seq_length

    def embed_documents(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)

    def embed_query(self, query: str) -> np.ndarray:
        prefix = QUERY_INSTRUCTION if self.use_query_instruction else ""
        return self.model.encode(prefix + query, normalize_embeddings=True, convert_to_numpy=True)
